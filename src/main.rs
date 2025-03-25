use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::{Arc, Mutex},
    time::Duration,
};

use rapier2d::{na::Isometry, prelude::*};

use foxglove::{
    schemas::{
        ArrowPrimitive, Color, FrameTransform, FrameTransforms, Quaternion, SceneEntity,
        SceneUpdate, Vector3,
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

#[derive(Default)]
struct Asteroids {
    clients: BTreeMap<u32, AsteroidClient>,
    gravity: Vector<f32>,
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
}

impl foxglove::websocket::ServerListener for AsteroidListener {
    fn on_client_advertise(
        &self,
        client: foxglove::websocket::Client,
        channel: &foxglove::websocket::ClientChannel,
    ) {
        println!("new client: {client:?}, channel: {channel:?}");
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

const UP: u32 = 1;
const LEFT: u32 = 2;
const DOWN: u32 = 4;
const RIGHT: u32 = 8;

const ROT_SPEED: f32 = 0.1;
const ACCEL: f32 = 0.1;
const ROT_DAMP: f32 = 5.0;
const BRAKE: f32 = 5.0;

impl AsteroidListener {
    fn new() -> Self {
        Self {
            asteroids: Mutex::new(Asteroids {
                clients: BTreeMap::new(),
                gravity: Vector::new(0.0, 0.0),
                integration_parameters: IntegrationParameters::default(),
                physics_pipeline: PhysicsPipeline::new(),
                island_manager: IslandManager::new(),
                broad_phase: DefaultBroadPhase::new(),
                narrow_phase: NarrowPhase::new(),
                impulse_joint_set: ImpulseJointSet::new(),
                multibody_joint_set: MultibodyJointSet::new(),
                ccd_solver: CCDSolver::new(),
                collider_set: ColliderSet::new(),
                rigid_body_set: RigidBodySet::new(),
            }),
        }
    }

    fn tick(&self) {
        // always forward-calculate the physics
        let mut state = self.asteroids.lock().unwrap();
        let asteroids = &mut *state;
        for (_, client) in asteroids.clients.iter_mut() {
            let body_handle = &mut asteroids.rigid_body_set[client.body_handle];
            if client.keys_pressed & UP != 0 {
                let rot = body_handle.rotation();
                body_handle.apply_impulse(Vector::new(rot.re * ACCEL, rot.im * ACCEL), true);
            }
            if client.keys_pressed & LEFT != 0 {
                body_handle.apply_torque_impulse(ROT_SPEED, true);
            }
            if client.keys_pressed & RIGHT != 0 {
                body_handle.apply_torque_impulse(-ROT_SPEED, true);
            }
            if client.keys_pressed & DOWN != 0 {
                body_handle.set_linear_damping(BRAKE);
            } else {
                body_handle.set_linear_damping(0.0);
            }

            let pos = body_handle.translation();
            if pos.x > WORLD_BOUND
                || pos.x < WORLD_BOUND
                || pos.y > WORLD_BOUND
                || pos.y < WORLD_BOUND
            {
                let new_x = ((pos.x + WORLD_BOUND) % (WORLD_BOUND * 2.0)) - WORLD_BOUND;
                let new_y = ((pos.y + WORLD_BOUND) % (WORLD_BOUND * 2.0)) - WORLD_BOUND;
                body_handle.set_position(
                    Isometry {
                        rotation: *body_handle.rotation(),
                        translation: Translation {
                            vector: Vector::new(new_x, new_y),
                        },
                    },
                    true,
                );
            }
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

        let entities = state
            .clients
            .iter()
            .map(|(&id, _)| SceneEntity {
                id: id.to_string(),
                frame_id: format!("player_{id}"),
                arrows: vec![ArrowPrimitive {
                    pose: None,
                    shaft_diameter: 0.1,
                    shaft_length: 0.5,
                    head_diameter: 0.15,
                    head_length: 0.1,
                    color: Some(Color {
                        r: 0.1,
                        g: 0.4,
                        b: 0.8,
                        a: 1.0,
                    }),
                }],
                ..Default::default()
            })
            .collect();
        BOXES.log(&SceneUpdate {
            deletions: Vec::new(),
            entities,
        });

        let transforms = state
            .clients
            .iter()
            .map(|(&id, client)| {
                let body = &state.rigid_body_set[client.body_handle];
                let pos = body.translation();
                let rot = body.rotation();
                let angle = rot.angle() / 2.0;
                FrameTransform {
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
                }
            })
            .collect();

        TF.log(&FrameTransforms { transforms });
        // every 2hz, publish the boxes
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
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().name = name.into();
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
}

fn make_player_body_handle(
    collider_set: &mut ColliderSet,
    rigid_body_set: &mut RigidBodySet,
) -> RigidBodyHandle {
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, 0.0])
        .angular_damping(ROT_DAMP)
        .build();
    let collider = ColliderBuilder::cuboid(0.5, 0.5).build();
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
