use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::{Arc, Mutex},
    time::{Duration, UNIX_EPOCH},
};

use rapier2d::{na::Isometry, prelude::*};

use foxglove::{
    schemas::{
        Color, CubePrimitive, FrameTransform, FrameTransforms, Log, ModelPrimitive, Pose,
        Quaternion, SceneEntity, SceneEntityDeletion, SceneUpdate, SpherePrimitive, TextPrimitive,
        Timestamp, Vector3,
    },
    websocket::{Capability, Client, ClientId},
};

const WORLD_BOUND: f32 = 5.0;

#[derive(Default)]
struct AsteroidListener {
    asteroids: Mutex<Asteroids>,
}

#[derive(Default)]
struct AsteroidClient {
    name: String,
    keys_pressed: u32,
    body_handle: RigidBodyHandle,
    last_shot_tick: Option<u64>,
}

struct Bullet {
    handle: RigidBodyHandle,
    born_tick: u64,
    alive: bool,
}

fn make_rock(collider_set: &mut ColliderSet, rigid_body_set: &mut RigidBodySet) -> RigidBodyHandle {
    let x = rand::random_range(-WORLD_BOUND..WORLD_BOUND);
    let y = rand::random_range(-WORLD_BOUND..WORLD_BOUND);
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![x, y])
        .angular_damping(ROT_DAMP)
        .rotation(rand::random_range(0.0..10.0))
        .build();
    let collider = ColliderBuilder::cuboid(0.25, 0.25).build();
    let body_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(collider, body_handle, rigid_body_set);
    body_handle
}

fn log(message: String) {
    let now = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap();
    let nanos = now.as_nanos();
    let nsec = (nanos % 1_000_000_000) as u32;
    let sec = (nanos / 1_000_000_000) as u32;
    LOGS.log(&Log {
        timestamp: Some(Timestamp::new(sec, nsec)),
        level: 2,
        name: "system".into(),
        message,
        ..Default::default()
    })
}

#[derive(Default)]
struct Asteroids {
    clients: BTreeMap<u32, AsteroidClient>,
    gravity: Vector<f32>,
    rocks: Vec<RigidBodyHandle>,
    bullets: Vec<Bullet>,
    next_bullet_idx: usize,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    collider_set: ColliderSet,
    rigid_body_set: RigidBodySet,
    tick_number: u64,
}

impl foxglove::websocket::ServerListener for AsteroidListener {
    fn on_client_unadvertise(
        &self,
        client: foxglove::websocket::Client,
        _channel: &foxglove::websocket::ClientChannel,
    ) {
        self.remove_client(client.id());
    }

    fn on_message_data(
        &self,
        client: foxglove::websocket::Client,
        client_channel: &foxglove::websocket::ClientChannel,
        payload: &[u8],
    ) {
        let id = client.id();
        if client_channel.topic == "/keys" {
            let stringnum = match std::str::from_utf8(payload) {
                Ok(n) => n,
                Err(err) => {
                    println!("client {id} sent non-utf8-formatted json: {err}");
                    return;
                }
            };
            let bitmap: u32 = match stringnum.parse() {
                Ok(n) => n,
                Err(err) => {
                    println!("client {id} sent JSON other than a number: {err}");
                    return;
                }
            };
            self.set_client_keys(id, bitmap);
        } else if client_channel.topic == "/my-name-is" {
            let nickname = match std::str::from_utf8(payload) {
                Ok(name) => name,
                Err(err) => {
                    println!("client {id} sent non-utf8-formatted json: {err}");
                    return;
                }
            };
            self.set_client_name(id, nickname);
        }
    }
}

foxglove::static_typed_channel!(BOXES, "/scene", foxglove::schemas::SceneUpdate);
foxglove::static_typed_channel!(TF, "/tf", foxglove::schemas::FrameTransforms);
foxglove::static_typed_channel!(LOGS, "/logs", foxglove::schemas::Log);

const UP: u32 = 1;
const LEFT: u32 = 2;
const DOWN: u32 = 4;
const RIGHT: u32 = 8;
const SHOOT: u32 = 16;

const BULLET_SPEED: f32 = 5.0;
const BULLET_LIFE: u64 = 100;
const ROT_SPEED: f32 = 0.005;
const ACCEL: f32 = 0.025;
const ROT_DAMP: f32 = 5.0;
const BRAKE: f32 = 5.0;

const SCENE_ENTITY_PBLISH_PERIOD: u64 = 10;
const FIRE_PERIOD: u64 = 5;

fn wrap_pos(body: &mut RigidBody) {
    let mut pos = *body.translation();
    let mut modified = false;
    if pos.x > WORLD_BOUND {
        pos.x -= 2. * WORLD_BOUND;
        modified = true;
    }
    if pos.x < -WORLD_BOUND {
        pos.x += 2. * WORLD_BOUND;
        modified = true;
    }
    if pos.y > WORLD_BOUND {
        pos.y -= 2. * WORLD_BOUND;
        modified = true;
    }
    if pos.y < -WORLD_BOUND {
        pos.y += 2. * WORLD_BOUND;
        modified = true;
    }
    if modified {
        body.set_position(
            Isometry {
                rotation: *body.rotation(),
                translation: Translation {
                    vector: Vector::new(pos.x, pos.y),
                },
            },
            true,
        );
    }
}

fn shoot(
    tick: u64,
    bullets: &mut Vec<Bullet>,
    next_bullet_idx: &mut usize,
    collider_set: &mut ColliderSet,
    rigid_body_set: &mut RigidBodySet,
    me: RigidBodyHandle,
) {
    let (my_pos, my_rot) = {
        let body = &rigid_body_set[me];
        (body.translation(), body.rotation())
    };
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![
            my_pos.x + (my_rot.re * 0.5),
            my_pos.y + (my_rot.im * 0.5)
        ])
        .linvel(Vector::new(
            my_rot.re * BULLET_SPEED,
            my_rot.im * BULLET_SPEED,
        ))
        .build();
    let collider = ColliderBuilder::ball(0.05).build();
    let body_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(collider, body_handle, rigid_body_set);
    let bullet = &mut bullets[*next_bullet_idx];
    bullet.alive = true;
    bullet.born_tick = tick;
    bullet.handle = body_handle;
    *next_bullet_idx += 1;
}

impl AsteroidListener {
    fn new() -> Self {
        let mut collider_set = ColliderSet::new();
        let mut rigid_body_set = RigidBodySet::new();
        let mut rocks = Vec::with_capacity(10);
        let mut bullets = Vec::with_capacity(1000);
        for _ in 0..10 {
            rocks.push(make_rock(&mut collider_set, &mut rigid_body_set));
        }
        for _ in 0..1000 {
            bullets.push(Bullet {
                born_tick: 0,
                alive: false,
                handle: RigidBodyHandle::invalid(),
            })
        }
        Self {
            asteroids: Mutex::new(Asteroids {
                clients: BTreeMap::new(),
                rocks,
                bullets,
                next_bullet_idx: 0,
                gravity: Vector::new(0.0, 0.0),
                integration_parameters: IntegrationParameters::default(),
                physics_pipeline: PhysicsPipeline::new(),
                island_manager: IslandManager::new(),
                broad_phase: DefaultBroadPhase::new(),
                narrow_phase: NarrowPhase::new(),
                impulse_joint_set: ImpulseJointSet::new(),
                multibody_joint_set: MultibodyJointSet::new(),
                ccd_solver: CCDSolver::new(),
                collider_set,
                rigid_body_set,
                tick_number: 0,
            }),
        }
    }

    fn tick(&self) {
        // always forward-calculate the physics
        let mut state = self.asteroids.lock().unwrap();
        let asteroids = &mut *state;
        for (_, client) in asteroids.clients.iter_mut() {
            let body = &mut asteroids.rigid_body_set[client.body_handle];
            if client.keys_pressed & UP != 0 {
                let rot = body.rotation();
                body.apply_impulse(Vector::new(rot.re * ACCEL, rot.im * ACCEL), true);
            }
            if client.keys_pressed & LEFT != 0 {
                body.apply_torque_impulse(ROT_SPEED, true);
            }
            if client.keys_pressed & RIGHT != 0 {
                body.apply_torque_impulse(-ROT_SPEED, true);
            }
            if client.keys_pressed & DOWN != 0 {
                body.set_linear_damping(BRAKE);
            } else {
                body.set_linear_damping(0.0);
            }
            wrap_pos(body);
            if client.keys_pressed & SHOOT != 0 {
                match client.last_shot_tick {
                    Some(tick) if tick + FIRE_PERIOD >= asteroids.tick_number => {}
                    _ => {
                        log(format!("{} shot!", client.name));
                        shoot(
                            asteroids.tick_number,
                            &mut asteroids.bullets,
                            &mut asteroids.next_bullet_idx,
                            &mut asteroids.collider_set,
                            &mut asteroids.rigid_body_set,
                            client.body_handle,
                        );
                        client.last_shot_tick = Some(asteroids.tick_number);
                    }
                }
            }
        }

        for handle in asteroids.rocks.iter() {
            let body = &mut asteroids.rigid_body_set[*handle];
            wrap_pos(body);
        }

        for bullet in asteroids.bullets.iter_mut() {
            if bullet.alive && bullet.born_tick + BULLET_LIFE < asteroids.tick_number {
                asteroids.rigid_body_set.remove(
                    bullet.handle,
                    &mut asteroids.island_manager,
                    &mut asteroids.collider_set,
                    &mut asteroids.impulse_joint_set,
                    &mut asteroids.multibody_joint_set,
                    true,
                );
                log(format!("bullet {:?} died", bullet.handle));
                bullet.alive = false;
            }
        }
        let (collision_send, collision_recv) = crossbeam::channel::unbounded();
        let (contact_force_send, _) = crossbeam::channel::unbounded();
        let event_handler = ChannelEventCollector::new(collision_send, contact_force_send);

        asteroids.physics_pipeline.step(
            &asteroids.gravity,
            &mut asteroids.integration_parameters,
            &mut asteroids.island_manager,
            &mut asteroids.broad_phase,
            &mut asteroids.narrow_phase,
            &mut asteroids.rigid_body_set,
            &mut asteroids.collider_set,
            &mut asteroids.impulse_joint_set,
            &mut asteroids.multibody_joint_set,
            &mut asteroids.ccd_solver,
            None,
            &(),
            &event_handler,
        );
        while let Ok(collision_event) = collision_recv.try_recv() {
            // Handle the collision event.
            let body1 = asteroids
                .collider_set
                .get(collision_event.collider1())
                .unwrap()
                .parent();
            let body2 = asteroids
                .collider_set
                .get(collision_event.collider2())
                .unwrap()
                .parent();
            for client in asteroids.clients.values() {
                let mut did_bonk = false;
                if let Some(handle) = body1 {
                    if client.body_handle == handle {
                        did_bonk = true;
                    }
                }
                if let Some(handle) = body2 {
                    if client.body_handle == handle {
                        did_bonk = true;
                    }
                }
                if did_bonk {
                    let name = &client.name;
                    log(format!("{name} hit something!"));
                }
            }
        }

        if asteroids.tick_number % SCENE_ENTITY_PBLISH_PERIOD == 0 {
            let mut entities = Vec::new();

            for (id, client) in asteroids.clients.iter() {
                entities.push(SceneEntity {
                    id: format!("player_{id}"),
                    frame_id: format!("player_{id}"),
                    models: vec![ModelPrimitive {
                        pose: Some(Pose {
                            position: None,
                            orientation: Some(Quaternion {
                                x: 0.0,
                                y: 0.0,
                                z: -0.707,
                                w: 0.707,
                            }),
                        }),
                        color: Some(Color {
                            r: 0.1,
                            g: 0.4,
                            b: 0.8,
                            a: 1.0,
                        }),
                        scale: Some(Vector3 {
                            x: 0.25,
                            y: 0.25,
                            z: 0.25,
                        }),
                        // override_color: true,
                        url: "package://ship.glb".into(),
                        ..Default::default()
                    }],
                    texts: vec![TextPrimitive {
                        billboard: true,
                        font_size: 0.2,
                        text: format!("\n\n\n\n{}", client.name),
                        ..Default::default()
                    }],
                    ..Default::default()
                });
            }
            for handle in asteroids.rocks.iter() {
                entities.push(SceneEntity {
                    id: format!("rock_{handle:?}"),
                    frame_id: format!("rock_{handle:?}"),
                    cubes: vec![CubePrimitive {
                        pose: None,
                        size: Some(Vector3 {
                            x: 0.5,
                            y: 0.5,
                            z: 0.5,
                        }),
                        color: Some(Color {
                            r: 0.5,
                            g: 0.5,
                            b: 0.5,
                            a: 1.0,
                        }),
                    }],
                    ..Default::default()
                })
            }
            for (idx, _) in asteroids.bullets.iter().enumerate() {
                entities.push(SceneEntity {
                    id: format!("bullet_{idx}"),
                    frame_id: format!("bullet_{idx}"),
                    spheres: vec![SpherePrimitive {
                        pose: None,
                        size: Some(Vector3 {
                            x: 0.05,
                            y: 0.05,
                            z: 0.05,
                        }),
                        color: Some(Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                    }],
                    ..Default::default()
                })
            }
            BOXES.log(&SceneUpdate {
                deletions: Vec::new(),
                entities,
            });
        }

        let mut transforms = Vec::new();
        for (id, client) in asteroids.clients.iter() {
            let body = &asteroids.rigid_body_set[client.body_handle];
            let pos = body.translation();
            let rot = body.rotation();
            let angle = rot.angle() / 2.0;
            transforms.push(FrameTransform {
                parent_frame_id: "world".into(),
                child_frame_id: format!("player_{id}"),
                translation: Some(Vector3 {
                    x: pos.x as _,
                    y: pos.y as _,
                    z: 0.0,
                }),
                rotation: Some(Quaternion {
                    x: 0.,
                    y: 0.,
                    z: angle.sin() as _,
                    w: angle.cos() as _, // real component
                }),
                ..Default::default()
            })
        }
        for handle in asteroids.rocks.iter() {
            let body = &asteroids.rigid_body_set[*handle];
            let pos = body.translation();
            let angle = body.rotation().angle() / 2.0;
            transforms.push(FrameTransform {
                parent_frame_id: "world".into(),
                child_frame_id: format!("rock_{handle:?}"),
                translation: Some(Vector3 {
                    x: pos.x as _,
                    y: pos.y as _,
                    z: 0.0,
                }),
                rotation: Some(Quaternion {
                    x: 0.,
                    y: 0.,
                    z: angle.sin() as _,
                    w: angle.cos() as _, // real component
                }),
                ..Default::default()
            });
        }
        for (idx, bullet) in asteroids.bullets.iter().enumerate() {
            let (x, y) = if bullet.alive {
                let handle = bullet.handle;
                let body = &asteroids.rigid_body_set[handle];
                let pos = body.translation();
                (pos.x, pos.y)
            } else {
                (99999.0, 99999.0)
            };
            transforms.push(FrameTransform {
                parent_frame_id: "world".into(),
                child_frame_id: format!("bullet_{idx}"),
                translation: Some(Vector3 {
                    x: x as _,
                    y: y as _,
                    z: 0.0,
                }),
                ..Default::default()
            });
        }

        TF.log(&FrameTransforms { transforms });
        asteroids.tick_number += 1;
    }

    fn set_client_name(&self, client_id: ClientId, name: &str) {
        let mut state = self.asteroids.lock().unwrap();
        let asteroids = &mut *state;
        let clients = &mut asteroids.clients;
        let collider_set = &mut asteroids.collider_set;
        let rigid_body_set = &mut asteroids.rigid_body_set;

        match clients.entry(client_id.into()) {
            Entry::Vacant(vacant) => {
                let body_handle = make_player_body_handle(collider_set, rigid_body_set);
                vacant.insert(AsteroidClient {
                    name: name.into(),
                    keys_pressed: 0,
                    body_handle,
                    last_shot_tick: None,
                });
                log(format!("Welcome {name}"));
            }
            Entry::Occupied(mut entry) => {
                if entry.get().name != name {
                    entry.get_mut().name = name.into();
                    log(format!("Welcome {name}"));
                }
            }
        }
    }
    fn set_client_keys(&self, client_id: ClientId, keys: u32) {
        let mut state = self.asteroids.lock().unwrap();
        let asteroids = &mut *state;
        let clients = &mut asteroids.clients;
        let collider_set = &mut asteroids.collider_set;
        let rigid_body_set = &mut asteroids.rigid_body_set;

        match clients.entry(client_id.into()) {
            Entry::Vacant(vacant) => {
                let body_handle = make_player_body_handle(collider_set, rigid_body_set);
                vacant.insert(AsteroidClient {
                    name: "(nameless)".into(),
                    keys_pressed: keys,
                    body_handle,
                    last_shot_tick: None,
                });
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().keys_pressed = keys;
            }
        }
    }
    fn remove_client(&self, client_id: ClientId) {
        let mut state = self.asteroids.lock().unwrap();
        let asteroids = &mut *state;
        let client_id: u32 = client_id.into();
        if let Some(client) = asteroids.clients.remove(&client_id) {
            let name = client.name;
            BOXES.log(&SceneUpdate {
                deletions: vec![SceneEntityDeletion {
                    timestamp: None,
                    r#type: 1,
                    id: format!("player_{client_id}"),
                }],
                entities: Vec::new(),
            });
            log(format!("Goodbye {name}!"));
        }
    }
}

fn make_player_body_handle(
    collider_set: &mut ColliderSet,
    rigid_body_set: &mut RigidBodySet,
) -> RigidBodyHandle {
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, 0.0])
        .angular_damping(ROT_DAMP)
        .build();
    let collider = ColliderBuilder::cuboid(0.25, 0.25)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    let body_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(collider, body_handle, rigid_body_set);
    body_handle
}

const SHIP: &[u8] = include_bytes!("ship.glb");

async fn handle_assets(_client: Client, path: String) -> Result<bytes::Bytes, String> {
    if path == "package://ship.glb" {
        return Ok(SHIP.into());
    } else {
        return Err(format!("{path} not found"));
    }
}

#[tokio::main]
async fn main() {
    let asteroids = Arc::new(AsteroidListener::new());
    let server = foxglove::WebSocketServer::new()
        .name("asteroid-server")
        .bind("0.0.0.0", 9999)
        .listener(asteroids.clone())
        .supported_encodings(["json"])
        .capabilities([Capability::ClientPublish])
        .fetch_asset_handler_async_fn(handle_assets)
        .start()
        .await
        .expect("Failed to start visualization server");
    println!("server started on localhost:9999");
    let mut interval = tokio::time::interval(Duration::from_millis(10));
    loop {
        tokio::select! {
            _ = interval.tick() => {
                asteroids.tick();
            },
            res = tokio::signal::ctrl_c() => {
                res.expect("failed to wait for sigint");
                server.stop().await;
                println!("stopped");
                break;
            }
        }
    }
}
