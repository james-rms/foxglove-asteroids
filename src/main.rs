use core::f32;
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::{Arc, Mutex},
    time::Duration,
};

use rapier2d::{na::Isometry, prelude::*};

use foxglove::{
    schemas::{
        Color, CubePrimitive, FrameTransform, FrameTransforms, Log, ModelPrimitive,
        PackedElementField, PointCloud, Pose, Quaternion, SceneEntity, SceneEntityDeletion,
        SceneUpdate, SpherePrimitive, TextPrimitive, Timestamp, Vector3,
    },
    websocket::{Capability, Client, ClientId},
};

const WORLD_BOUND: f32 = 5.0;

#[derive(Default)]
struct AsteroidListener {
    asteroids: Mutex<Asteroids>,
}

struct Player {
    name: String,
    keys_pressed: u32,
    last_seen: u64,
    state: PlayerState,
}

enum PlayerState {
    Alive {
        handle: RigidBodyHandle,
        health: u32,
        last_shot_tick: Option<u64>,
    },
    Dead {
        tick: u64,
        x: f32,
        y: f32,
    },
}

#[derive(Clone, Copy)]
enum EntityType {
    Rock,
    Player,
    Bullet,
}

#[derive(Clone, Copy)]
struct BodyUserData {
    what: EntityType,
    id: u32,
}

impl BodyUserData {
    fn pack(self) -> u128 {
        let discriminant = match self.what {
            EntityType::Rock => 1,
            EntityType::Player => 2,
            EntityType::Bullet => 3,
        };
        (discriminant << 32) + (self.id as u128)
    }
    fn unpack(packed: u128) -> Self {
        let discriminant = packed >> 32;
        let id = (packed & ((1 << 32) - 1)) as u32;
        let what = match discriminant {
            1 => EntityType::Rock,
            2 => EntityType::Player,
            3 => EntityType::Bullet,
            _ => panic!("expected valid discriminant, got {discriminant}"),
        };
        Self { id, what }
    }
}

struct Bullet {
    handle: Option<RigidBodyHandle>,
    born_tick: u64,
}

enum Rock {
    Alive {
        handle: RigidBodyHandle,
        health: u32,
    },
    Dead {
        tick: u64,
        x: f32,
        y: f32,
    },
}

fn draw_death_rings(point_storage: &mut Vec<u8>, tick_delta: u64, x: f32, y: f32) {
    // want radius = 3.0 when tick_delta = 100
    if tick_delta > 100 {
        return;
    }
    let radius = (tick_delta as f32) * (0.03);
    for t in 0..16 {
        let theta = (f32::consts::PI / 8.0) * t as f32;
        let dx = theta.cos() * radius;
        let dy = theta.sin() * radius;
        let x = x + dx;
        let y = y + dy;
        let z: f32 = 0.0;
        point_storage.extend_from_slice(&x.to_le_bytes());
        point_storage.extend_from_slice(&y.to_le_bytes());
        point_storage.extend_from_slice(&z.to_le_bytes());
    }
}

fn make_rock(collider_set: &mut ColliderSet, rigid_body_set: &mut RigidBodySet, id: u32) -> Rock {
    let x = rand::random_range(-WORLD_BOUND..WORLD_BOUND);
    let y = rand::random_range(-WORLD_BOUND..WORLD_BOUND);
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![x, y])
        .angvel(rand::random_range(0.0..10.0))
        .linvel(Vector::new(
            rand::random_range(-2.0..2.0),
            rand::random_range(-2.0..2.0),
        ))
        .angular_damping(ROT_DAMP)
        .rotation(rand::random_range(0.0..10.0))
        .user_data(
            BodyUserData {
                id,
                what: EntityType::Rock,
            }
            .pack(),
        )
        .build();
    let collider = ColliderBuilder::cuboid(0.25, 0.25)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    let handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(collider, handle, rigid_body_set);
    Rock::Alive {
        handle,
        health: ROCK_INIT_HEALTH,
    }
}

fn log(message: String) {
    LOGS.log(&Log {
        timestamp: Some(Timestamp::try_from(std::time::SystemTime::now()).unwrap()),
        level: foxglove::schemas::log::Level::Info.into(),
        name: "system".into(),
        message,
        ..Default::default()
    })
}

#[derive(Default)]
struct Asteroids {
    clients: BTreeMap<u32, Player>,
    gravity: Vector<f32>,
    rocks: Vec<Rock>,
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
        let mut state = self.asteroids.lock().unwrap();
        let asteroids = &mut *state;
        let client_id: u32 = client.id().into();
        remove_player(asteroids, client_id);
    }

    fn on_message_data(
        &self,
        client: foxglove::websocket::Client,
        client_channel: &foxglove::websocket::ClientChannel,
        payload: &[u8],
    ) {
        let id = client.id();
        if client_channel.topic == "/keys" {
            let bitmap: u32 = match serde_json::from_slice(payload) {
                Ok(bitmap) => bitmap,
                Err(err) => {
                    println!("client {id} sent non-number key bitmap: {err}");
                    return;
                }
            };
            self.set_client_keys(id, bitmap);
        } else if client_channel.topic == "/my-name-is" {
            let nickname: String = match serde_json::from_slice(payload) {
                Ok(nickname) => nickname,
                Err(err) => {
                    println!("client {id} sent non-string nickname: {err}");
                    return;
                }
            };
            self.set_client_name(id, &nickname);
        }
    }
}

foxglove::static_typed_channel!(BOXES, "/scene", foxglove::schemas::SceneUpdate);
foxglove::static_typed_channel!(TF, "/tf", foxglove::schemas::FrameTransforms);
foxglove::static_typed_channel!(LOGS, "/logs", foxglove::schemas::Log);
foxglove::static_typed_channel!(POINTCLOUD, "/pointcloud", foxglove::schemas::PointCloud);

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
const ROCK_INIT_HEALTH: u32 = 5;
const NUM_ROCKS: u32 = 10;
const PLAYER_INIT_HEALTH: u32 = 10;

const SCENE_ENTITY_PUBLISH_PERIOD: u64 = 100;
const FIRE_PERIOD: u64 = 10;
const RESURRECTION_TICKS: u64 = 500;
const INACTIVE_KICK_TICKS: u64 = 2000;

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

fn ding_rock(rock: &mut Rock, rigid_body_set: &RigidBodySet, tick: u64) -> Option<RigidBodyHandle> {
    let res = match rock {
        Rock::Alive { handle, health: 0 } => {
            let body = &rigid_body_set[*handle];
            let pos = body.translation();
            Some((pos.x, pos.y, *handle))
        }
        Rock::Alive { health, .. } => {
            *health -= 1;
            None
        }
        Rock::Dead { .. } => None,
    };
    if let Some((x, y, handle)) = res {
        *rock = Rock::Dead { tick, x, y };
        Some(handle)
    } else {
        None
    }
}

fn ding_player(
    player: &mut Player,
    rigid_body_set: &mut RigidBodySet,
    tick: u64,
) -> Option<RigidBodyHandle> {
    let res = match &mut player.state {
        PlayerState::Alive {
            health: 0, handle, ..
        } => {
            log(format!("{} died, will resurrect in 5 seconds", player.name));
            let body = &rigid_body_set[*handle];
            let pos = body.translation();
            Some((pos.x, pos.y, *handle))
        }
        PlayerState::Alive { health, .. } => {
            *health -= 1;
            log(format!("{} down to {health} health", player.name));
            None
        }
        PlayerState::Dead { .. } => None,
    };
    if let Some((x, y, handle)) = res {
        player.state = PlayerState::Dead { tick, x, y };
        Some(handle)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn rock_player_collision(
    player: &mut Player,
    rock: &mut Rock,
    island_manager: &mut IslandManager,
    collider_set: &mut ColliderSet,
    rigid_body_set: &mut RigidBodySet,
    impulse_joint_set: &mut ImpulseJointSet,
    multibody_joint_set: &mut MultibodyJointSet,
    tick: u64,
) {
    log(format!("{} hit a rock", player.name));
    if let Some(handle) = ding_player(player, rigid_body_set, tick) {
        rigid_body_set.remove(
            handle,
            island_manager,
            collider_set,
            impulse_joint_set,
            multibody_joint_set,
            true,
        );
    };
    if let Some(handle) = ding_rock(rock, rigid_body_set, tick) {
        rigid_body_set.remove(
            handle,
            island_manager,
            collider_set,
            impulse_joint_set,
            multibody_joint_set,
            true,
        );
    };
}
fn remove_player(asteroids: &mut Asteroids, client_id: u32) {
    if let Some(client) = asteroids.clients.remove(&client_id) {
        let name = client.name;
        BOXES.log(&SceneUpdate {
            deletions: vec![SceneEntityDeletion {
                timestamp: None,
                r#type: foxglove::schemas::scene_entity_deletion::Type::MatchingId.into(),
                id: format!("player_{client_id}"),
            }],
            entities: Vec::new(),
        });
        log(format!("Goodbye {name}!"));
    }
}

fn shoot(
    tick: u64,
    bullets: &mut [Bullet],
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
        .user_data(
            BodyUserData {
                id: *next_bullet_idx as u32,
                what: EntityType::Bullet,
            }
            .pack(),
        )
        .build();
    let collider = ColliderBuilder::ball(0.1)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    let body_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(collider, body_handle, rigid_body_set);
    let bullet = &mut bullets[*next_bullet_idx];
    bullet.born_tick = tick;
    bullet.handle = Some(body_handle);

    *next_bullet_idx = (*next_bullet_idx + 1) % bullets.len();
}

impl AsteroidListener {
    fn new() -> Self {
        let mut collider_set = ColliderSet::new();
        let mut rigid_body_set = RigidBodySet::new();
        let mut rocks = Vec::with_capacity(10);
        let mut bullets = Vec::with_capacity(000);
        for idx in 0..NUM_ROCKS {
            rocks.push(make_rock(&mut collider_set, &mut rigid_body_set, idx));
        }
        for _ in 0..100 {
            bullets.push(Bullet {
                born_tick: 0,
                handle: None,
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
        let mut state = self.asteroids.lock().unwrap();
        let asteroids = &mut *state;

        let dead_player_ids: Vec<(u32, String)> = asteroids
            .clients
            .iter()
            .filter_map(|(id, player)| {
                if player.last_seen + INACTIVE_KICK_TICKS < asteroids.tick_number {
                    Some((*id, player.name.clone()))
                } else {
                    None
                }
            })
            .collect();

        for (id, name) in dead_player_ids {
            log(format!("removing {name} due to inactivity"));
            remove_player(asteroids, id);
        }

        for (id, player) in asteroids.clients.iter_mut() {
            match &mut player.state {
                PlayerState::Alive {
                    handle,
                    last_shot_tick,
                    ..
                } => {
                    let body = &mut asteroids.rigid_body_set[*handle];
                    if player.keys_pressed & UP != 0 {
                        let rot = body.rotation();
                        body.apply_impulse(Vector::new(rot.re * ACCEL, rot.im * ACCEL), true);
                    }
                    if player.keys_pressed & LEFT != 0 {
                        body.apply_torque_impulse(ROT_SPEED, true);
                    }
                    if player.keys_pressed & RIGHT != 0 {
                        body.apply_torque_impulse(-ROT_SPEED, true);
                    }
                    if player.keys_pressed & DOWN != 0 {
                        body.set_linear_damping(BRAKE);
                    } else {
                        body.set_linear_damping(0.0);
                    }
                    wrap_pos(body);
                    if player.keys_pressed & SHOOT != 0 {
                        match *last_shot_tick {
                            Some(tick) if tick + FIRE_PERIOD >= asteroids.tick_number => {}
                            _ => {
                                shoot(
                                    asteroids.tick_number,
                                    &mut asteroids.bullets,
                                    &mut asteroids.next_bullet_idx,
                                    &mut asteroids.collider_set,
                                    &mut asteroids.rigid_body_set,
                                    *handle,
                                );
                                *last_shot_tick = Some(asteroids.tick_number);
                            }
                        }
                    }
                }
                PlayerState::Dead { tick, .. } => {
                    if asteroids.tick_number > *tick + RESURRECTION_TICKS {
                        log(format!("{} is resurrected!", player.name));
                        player.state = PlayerState::Alive {
                            handle: make_player_body_handle(
                                &mut asteroids.collider_set,
                                &mut asteroids.rigid_body_set,
                                *id,
                            ),
                            health: PLAYER_INIT_HEALTH,
                            last_shot_tick: None,
                        }
                    }
                }
            }
        }

        for rock in asteroids.rocks.iter() {
            if let Rock::Alive { handle, .. } = rock {
                let body = &mut asteroids.rigid_body_set[*handle];
                wrap_pos(body);
            }
        }
        if asteroids
            .rocks
            .iter()
            .filter(|&r| matches!(r, Rock::Alive { .. }))
            .count()
            == 0
        {
            log("All rocks destroyed, well done everybody!".into());
            log("New game starts now!".into());
            asteroids.rocks.clear();
            for id in 0..NUM_ROCKS {
                asteroids.rocks.push(make_rock(
                    &mut asteroids.collider_set,
                    &mut asteroids.rigid_body_set,
                    id,
                ));
            }
        }
        for rock in asteroids.rocks.iter() {
            if let Rock::Alive { handle, .. } = rock {
                let body = &mut asteroids.rigid_body_set[*handle];
                wrap_pos(body);
            }
        }

        for bullet in asteroids.bullets.iter_mut() {
            if let Some(handle) = bullet.handle {
                let body = &mut asteroids.rigid_body_set[handle];
                wrap_pos(body);
                if bullet.born_tick + BULLET_LIFE < asteroids.tick_number {
                    asteroids.rigid_body_set.remove(
                        handle,
                        &mut asteroids.island_manager,
                        &mut asteroids.collider_set,
                        &mut asteroids.impulse_joint_set,
                        &mut asteroids.multibody_joint_set,
                        true,
                    );
                    bullet.handle = None;
                }
            }
        }
        let (collision_send, collision_recv) = crossbeam::channel::unbounded();
        let (contact_force_send, _) = crossbeam::channel::unbounded();
        let event_handler = ChannelEventCollector::new(collision_send, contact_force_send);

        asteroids.physics_pipeline.step(
            &asteroids.gravity,
            &asteroids.integration_parameters,
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
            let Some(entity1) = asteroids
                .collider_set
                .get(collision_event.collider1())
                .and_then(|collider| collider.parent())
                .map(|handle| BodyUserData::unpack(asteroids.rigid_body_set[handle].user_data))
            else {
                continue;
            };
            let Some(entity2) = asteroids
                .collider_set
                .get(collision_event.collider2())
                .and_then(|collider| collider.parent())
                .map(|handle| BodyUserData::unpack(asteroids.rigid_body_set[handle].user_data))
            else {
                continue;
            };
            match (entity1.what, entity2.what) {
                (EntityType::Rock, EntityType::Rock) => {}
                (EntityType::Bullet, EntityType::Bullet) => {}
                (EntityType::Player, EntityType::Rock) => {
                    let Some(player) = asteroids.clients.get_mut(&entity1.id) else {
                        continue;
                    };
                    let rock = &mut asteroids.rocks[entity2.id as usize];
                    rock_player_collision(
                        player,
                        rock,
                        &mut asteroids.island_manager,
                        &mut asteroids.collider_set,
                        &mut asteroids.rigid_body_set,
                        &mut asteroids.impulse_joint_set,
                        &mut asteroids.multibody_joint_set,
                        asteroids.tick_number,
                    );
                }
                (EntityType::Rock, EntityType::Player) => {
                    let rock = &mut asteroids.rocks[entity1.id as usize];
                    let Some(player) = asteroids.clients.get_mut(&entity2.id) else {
                        continue;
                    };
                    rock_player_collision(
                        player,
                        rock,
                        &mut asteroids.island_manager,
                        &mut asteroids.collider_set,
                        &mut asteroids.rigid_body_set,
                        &mut asteroids.impulse_joint_set,
                        &mut asteroids.multibody_joint_set,
                        asteroids.tick_number,
                    );
                }
                (EntityType::Player, EntityType::Player) => {
                    let Some(player) = asteroids.clients.get_mut(&entity1.id) else {
                        continue;
                    };
                    if let Some(handle) =
                        ding_player(player, &mut asteroids.rigid_body_set, asteroids.tick_number)
                    {
                        asteroids.rigid_body_set.remove(
                            handle,
                            &mut asteroids.island_manager,
                            &mut asteroids.collider_set,
                            &mut asteroids.impulse_joint_set,
                            &mut asteroids.multibody_joint_set,
                            true,
                        );
                    }
                    let Some(player) = asteroids.clients.get_mut(&entity2.id) else {
                        continue;
                    };
                    if let Some(handle) =
                        ding_player(player, &mut asteroids.rigid_body_set, asteroids.tick_number)
                    {
                        asteroids.rigid_body_set.remove(
                            handle,
                            &mut asteroids.island_manager,
                            &mut asteroids.collider_set,
                            &mut asteroids.impulse_joint_set,
                            &mut asteroids.multibody_joint_set,
                            true,
                        );
                    }
                }
                (EntityType::Player, EntityType::Bullet) => {
                    let Some(player) = asteroids.clients.get_mut(&entity1.id) else {
                        continue;
                    };
                    if let Some(handle) =
                        ding_player(player, &mut asteroids.rigid_body_set, asteroids.tick_number)
                    {
                        asteroids.rigid_body_set.remove(
                            handle,
                            &mut asteroids.island_manager,
                            &mut asteroids.collider_set,
                            &mut asteroids.impulse_joint_set,
                            &mut asteroids.multibody_joint_set,
                            true,
                        );
                    }
                }
                (EntityType::Bullet, EntityType::Player) => {
                    let Some(player) = asteroids.clients.get_mut(&entity2.id) else {
                        continue;
                    };
                    if let Some(handle) =
                        ding_player(player, &mut asteroids.rigid_body_set, asteroids.tick_number)
                    {
                        asteroids.rigid_body_set.remove(
                            handle,
                            &mut asteroids.island_manager,
                            &mut asteroids.collider_set,
                            &mut asteroids.impulse_joint_set,
                            &mut asteroids.multibody_joint_set,
                            true,
                        );
                    }
                }
                (EntityType::Rock, EntityType::Bullet) => {
                    if let Some(handle) = ding_rock(
                        &mut asteroids.rocks[entity1.id as usize],
                        &asteroids.rigid_body_set,
                        asteroids.tick_number,
                    ) {
                        asteroids.rigid_body_set.remove(
                            handle,
                            &mut asteroids.island_manager,
                            &mut asteroids.collider_set,
                            &mut asteroids.impulse_joint_set,
                            &mut asteroids.multibody_joint_set,
                            true,
                        );
                    }
                }
                (EntityType::Bullet, EntityType::Rock) => {
                    if let Some(handle) = ding_rock(
                        &mut asteroids.rocks[entity2.id as usize],
                        &asteroids.rigid_body_set,
                        asteroids.tick_number,
                    ) {
                        asteroids.rigid_body_set.remove(
                            handle,
                            &mut asteroids.island_manager,
                            &mut asteroids.collider_set,
                            &mut asteroids.impulse_joint_set,
                            &mut asteroids.multibody_joint_set,
                            true,
                        );
                    }
                    // ditto
                }
            }
        }

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
            for (idx, _) in asteroids.rocks.iter().enumerate() {
                entities.push(SceneEntity {
                    id: format!("rock_{idx}"),
                    frame_id: format!("rock_{idx}"),
                    cubes: vec![CubePrimitive {
                        pose: None,
                        size: Some(Vector3 {
                            x: 0.5,
                            y: 0.5,
                            z: 0.5,
                        }),
                        color: Some(Color {
                            r: 0.5,
                            g: 0.7,
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
                            x: 0.1,
                            y: 0.1,
                            z: 0.1,
                        }),
                        color: Some(Color {
                            r: 1.0,
                            g: 0.3,
                            b: 0.3,
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

        let mut pointcloud_storage = Vec::new();
        for player in asteroids.clients.values() {
            if let PlayerState::Dead { tick, x, y } = &player.state {
                let tick_delta = asteroids.tick_number - *tick;
                draw_death_rings(&mut pointcloud_storage, tick_delta, *x, *y);
            }
        }
        for rock in asteroids.rocks.iter() {
            if let Rock::Dead { tick, x, y } = rock {
                let tick_delta = asteroids.tick_number - *tick;
                draw_death_rings(&mut pointcloud_storage, tick_delta, *x, *y);
            }
        }
        POINTCLOUD.log(&PointCloud {
            timestamp: Some(Timestamp::try_from(std::time::SystemTime::now()).unwrap()),
            frame_id: "world".into(),
            point_stride: 12,
            fields: vec![
                PackedElementField {
                    name: "x".into(),
                    offset: 0,
                    r#type: foxglove::schemas::packed_element_field::NumericType::Float32.into(),
                },
                PackedElementField {
                    name: "y".into(),
                    offset: 4,
                    r#type: foxglove::schemas::packed_element_field::NumericType::Float32.into(),
                },
                PackedElementField {
                    name: "z".into(),
                    offset: 8,
                    r#type: foxglove::schemas::packed_element_field::NumericType::Float32.into(),
                },
            ],
            data: pointcloud_storage.into(),
            pose: None,
        });

        let mut transforms = Vec::new();
        for (id, player) in asteroids.clients.iter() {
            let (x, y, qz, qw) = match player.state {
                PlayerState::Alive { handle, .. } => {
                    let body = &asteroids.rigid_body_set[handle];
                    let pos = body.translation();
                    let rot = body.rotation();
                    let angle = rot.angle() / 2.0;
                    (pos.x, pos.y, angle.sin(), angle.cos())
                }
                PlayerState::Dead { .. } => (9999.0, 9999.0, 1.0, 0.0),
            };
            transforms.push(FrameTransform {
                parent_frame_id: "world".into(),
                child_frame_id: format!("player_{id}"),
                translation: Some(Vector3 {
                    x: x as _,
                    y: y as _,
                    z: 0.0,
                }),
                rotation: Some(Quaternion {
                    x: 0.,
                    y: 0.,
                    z: qz as _,
                    w: qw as _, // real component
                }),
                ..Default::default()
            })
        }
        for (idx, rock) in asteroids.rocks.iter().enumerate() {
            let (x, y, qz, qw) = match rock {
                Rock::Alive { handle, .. } => {
                    let body = &asteroids.rigid_body_set[*handle];
                    let pos = body.translation();
                    let angle = body.rotation().angle() / 2.0;
                    (pos.x, pos.y, angle.sin(), angle.cos())
                }
                Rock::Dead { .. } => (9999.0, 9999.0, 1.0, 0.0),
            };
            transforms.push(FrameTransform {
                parent_frame_id: "world".into(),
                child_frame_id: format!("rock_{idx}"),
                translation: Some(Vector3 {
                    x: x as _,
                    y: y as _,
                    z: 0.0,
                }),
                rotation: Some(Quaternion {
                    x: 0.,
                    y: 0.,
                    z: qz as _,
                    w: qw as _, // real component
                }),
                ..Default::default()
            });
        }
        for (idx, bullet) in asteroids.bullets.iter().enumerate() {
            let (x, y) = if let Some(handle) = bullet.handle {
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
                let handle = make_player_body_handle(collider_set, rigid_body_set, *vacant.key());
                vacant.insert(Player {
                    name: name.into(),
                    keys_pressed: 0,
                    last_seen: asteroids.tick_number,
                    state: PlayerState::Alive {
                        handle,
                        health: PLAYER_INIT_HEALTH,
                        last_shot_tick: None,
                    },
                });
                log(format!("Welcome {name}"));
            }
            Entry::Occupied(mut entry) => {
                let player = entry.get_mut();
                if player.name != name {
                    player.name = name.into();
                    log(format!("Welcome {name}"));
                }
                player.last_seen = asteroids.tick_number;
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
                let handle = make_player_body_handle(collider_set, rigid_body_set, *vacant.key());
                vacant.insert(Player {
                    name: "(nameless)".into(),
                    keys_pressed: keys,
                    last_seen: asteroids.tick_number,
                    state: PlayerState::Alive {
                        handle,
                        health: PLAYER_INIT_HEALTH,
                        last_shot_tick: None,
                    },
                });
            }
            Entry::Occupied(mut entry) => {
                let player = entry.get_mut();
                player.keys_pressed = keys;
                player.last_seen = asteroids.tick_number;
            }
        }
    }
}

fn make_player_body_handle(
    collider_set: &mut ColliderSet,
    rigid_body_set: &mut RigidBodySet,
    id: u32,
) -> RigidBodyHandle {
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, 0.0])
        .angular_damping(ROT_DAMP)
        .user_data(
            BodyUserData {
                id,
                what: EntityType::Player,
            }
            .pack(),
        )
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
        Ok(SHIP.into())
    } else {
        Err(format!("{path} not found"))
    }
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
        .fetch_asset_handler_async_fn(handle_assets)
        .start()
        .await
        .expect("Failed to start visualization server");
    println!("server started on 127.0.0.1:9999");
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
