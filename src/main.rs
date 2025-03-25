use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::{Arc, Mutex},
    time::Duration,
};

use rapier2d::{na::Isometry, prelude::*};

use foxglove::{
    schemas::{
        Color, CubePrimitive, FrameTransform, FrameTransforms, Log, ModelPrimitive, Pose,
        Quaternion, SceneEntity, SceneEntityDeletion, SceneUpdate, TextPrimitive, Vector3,
    },
    websocket::{Capability, ClientId},
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

#[derive(Default)]
struct Asteroids {
    clients: BTreeMap<u32, AsteroidClient>,
    gravity: Vector<f32>,
    rocks: Vec<RigidBodyHandle>,
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

const ROT_SPEED: f32 = 0.005;
const ACCEL: f32 = 0.025;
const ROT_DAMP: f32 = 5.0;
const BRAKE: f32 = 5.0;

const SCENE_ENTITY_PUBLISH_PERIOD: u64 = 10;

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

impl AsteroidListener {
    fn new() -> Self {
        let mut collider_set = ColliderSet::new();
        let mut rigid_body_set = RigidBodySet::new();
        let mut rocks = Vec::with_capacity(10);
        for _ in 0..10 {
            rocks.push(make_rock(&mut collider_set, &mut rigid_body_set));
        }
        Self {
            asteroids: Mutex::new(Asteroids {
                clients: BTreeMap::new(),
                rocks,
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
        }

        for handle in asteroids.rocks.iter() {
            let body = &mut asteroids.rigid_body_set[*handle];
            wrap_pos(body);
        }

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
            &(),
        );

        if asteroids.tick_number % SCENE_ENTITY_PUBLISH_PERIOD == 0 {
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
                        url: "file:///Users/j/ship.glb".into(),
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
                });
                LOGS.log(&Log {
                    message: format!("Welcome {name}!"),
                    ..Default::default()
                });
            }
            Entry::Occupied(mut entry) => {
                if entry.get().name != name {
                    entry.get_mut().name = name.into();
                    LOGS.log(&Log {
                        message: format!("Welcome {name}!"),
                        ..Default::default()
                    });
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
            LOGS.log(&Log {
                message: format!("Goodbye {name}!"),
                ..Default::default()
            });
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
    let collider = ColliderBuilder::cuboid(0.25, 0.25).build();
    let body_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(collider, body_handle, rigid_body_set);
    body_handle
}

#[tokio::main]
async fn main() {
    let asteroids = Arc::new(AsteroidListener::new());
    let server = foxglove::WebSocketServer::new()
        .name("asteroid-server")
        .bind("127.0.0.1", 9999)
        .listener(asteroids.clone())
        .supported_encodings(["json"])
        .capabilities([Capability::ClientPublish])
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
