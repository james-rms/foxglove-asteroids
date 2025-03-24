use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use rapier2d::prelude::*;

use foxglove::{
    schemas::{
        ArrowPrimitive, Color, FrameTransform, FrameTransforms, Quaternion, SceneEntity,
        SceneUpdate, Vector3,
    },
    websocket::{Capability, ClientId},
};

const WORLD_BOUND: f64 = 5.0;

#[derive(Default)]
struct AsteroidListener {
    asteroids: Mutex<Asteroids>,
}

#[derive(Default)]
struct AsteroidClient {
    name: String,
    keys_pressed: u32,
    x: f64,
    dx: f64,
    y: f64,
    dy: f64,
    r: f64,
}

#[derive(Default)]
struct Asteroids {
    clients: BTreeMap<u32, AsteroidClient>,
    gravity: Vec<f64>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
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

const ROT_SPEED: f64 = 3.0;
const ACCEL: f64 = 3.0;
const BRAKE: f64 = 0.8;

impl AsteroidListener {
    fn new() -> Self {
        Self {
            asteroids: Mutex::new(Asteroids {
                clients: BTreeMap::new(),
                gravity: vec![0.0, 0.0],
                integration_parameters: IntegrationParameters::default(),
                physics_pipeline: PhysicsPipeline::new(),
                island_manager: IslandManager::new(),
                broad_phase: DefaultBroadPhase::new(),
                narrow_phase: NarrowPhase::new(),
                impulse_joint_set: ImpulseJointSet::new(),
                multibody_joint_set: MultibodyJointSet::new(),
                ccd_solver: CCDSolver::new(),
                query_pipeline: QueryPipeline::new(),
            }),
        }
    }

    async fn tick(&self, dt: Duration) {
        // always forward-calculate the physics
        let mut state = self.asteroids.lock().unwrap();
        let dt = dt.as_secs_f64();
        for (_, client) in state.clients.iter_mut() {
            if client.keys_pressed & UP != 0 {
                client.dx += client.r.cos() * ACCEL * dt;
                client.dy += client.r.sin() * ACCEL * dt;
            }
            if client.keys_pressed & LEFT != 0 {
                client.r += ROT_SPEED * dt;
            }
            if client.keys_pressed & RIGHT != 0 {
                client.r -= ROT_SPEED * dt;
            }
            if client.keys_pressed & DOWN != 0 {
                client.dx *= BRAKE;
                client.dy *= BRAKE;
            }
            client.x = client.x + client.dx * dt;
            client.y = client.y + client.dy * dt;

            while client.x > WORLD_BOUND {
                client.x -= WORLD_BOUND * 2.;
            }
            while client.x < -WORLD_BOUND {
                client.x += WORLD_BOUND * 2.;
            }
            while client.y > WORLD_BOUND {
                client.y -= WORLD_BOUND * 2.;
            }
            while client.y < -WORLD_BOUND {
                client.y += WORLD_BOUND * 2.;
            }
        }
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
            .map(|(&id, client)| FrameTransform {
                parent_frame_id: "world".into(),
                child_frame_id: format!("player_{id}"),
                translation: Some(Vector3 {
                    x: client.x,
                    y: client.y,
                    z: 0.0,
                }),
                rotation: Some(Quaternion {
                    x: 0.,
                    y: 0.,
                    z: (client.r / 2.).sin(),
                    w: (client.r / 2.).cos(),
                }),
                ..Default::default()
            })
            .collect();

        TF.log(&FrameTransforms { transforms });
        // every 2hz, publish the boxes
    }

    fn set_client_name(&self, client_id: ClientId, name: &str) {
        self.asteroids
            .lock()
            .unwrap()
            .clients
            .entry(client_id.into())
            .and_modify(|c| c.name = name.into())
            .or_insert(AsteroidClient {
                name: name.into(),
                ..Default::default()
            });
    }
    fn set_client_keys(&self, client_id: ClientId, keys: u32) {
        self.asteroids
            .lock()
            .unwrap()
            .clients
            .entry(client_id.into())
            .and_modify(|c| {
                c.keys_pressed = keys;
                let name = &c.name;
            })
            .or_insert(AsteroidClient {
                name: "(nameless)".into(),
                keys_pressed: keys,
                ..Default::default()
            });
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
        .start()
        .await
        .expect("Failed to start visualization server");
    println!("server started on localhost:9999");
    let mut interval = tokio::time::interval(Duration::from_millis(10));
    let mut last_tick_time = std::time::Instant::now();
    loop {
        tokio::select! {
            _ = interval.tick() => {
                asteroids.tick(last_tick_time.elapsed()).await;
                last_tick_time = std::time::Instant::now();
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
