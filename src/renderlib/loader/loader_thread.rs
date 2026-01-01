use crate::profile_span;
use crate::renderlib::messages::LoaderMessage;
use crate::renderlib::messages::RenderMessage;
use crate::renderlib::textures::raw_image::RawImage;
use crate::renderlib::user_event::UserEvent;
use std::sync::mpsc;
use std::thread;

use log::*;

use vulkanalia::vk;

use anyhow::{Result, anyhow};

struct LoaderThread {
    render_message_sender: mpsc::Sender<RenderMessage>,
    loader_message_reciever: mpsc::Receiver<LoaderMessage>,

    // TODO: Хардкод на winit для дерганья poll_messages и перерисовки в ивент лупе, нужна адекватная замена.
    event_loop_proxy: winit::event_loop::EventLoopProxy<UserEvent>,

    message_index: u64,
}

impl LoaderThread {
    fn run(&mut self) {
        while let Ok(cmd) = self.loader_message_reciever.recv() {
            let image_path = if self.message_index % 2 == 0 {
                "C:/RustProjects/vulkan-map/resources/texture.png"
            } else {
                "C:/RustProjects/vulkan-map/resources/tile.png"
            };
            match cmd {
                LoaderMessage::LoadImage => {
                    match self.load_image(image_path) {
                        Ok(raw) => {
                            self.render_message_sender
                                .send(RenderMessage::RawImage(raw))
                                .ok();

                            self.event_loop_proxy
                                .send_event(UserEvent::RequestRenderPoll)
                                .ok();
                        }
                        Err(err) => error!("Failed to load image: {}", err),
                    };
                }
                LoaderMessage::Stop => break,
            }
            self.message_index += 1;
        }
    }

    fn load_image(&self, image_path: &str) -> Result<RawImage> {
        let _span = profile_span!("Loader::load_image");

        // Загружаем изображение и сразу приводим к RGBA8
        let img = image::open(image_path)
            .map_err(|e| anyhow!("Failed to load image {}: {}", image_path, e))?
            .to_rgba8();

        let (width, height) = img.dimensions();
        let pixels = img.into_raw();

        let size_bytes = pixels.len() as u64;
        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        Ok(RawImage {
            width,
            height,
            format: vk::Format::R8G8B8A8_SRGB,
            pixels,
            size_bytes,
            mip_levels,
        })
    }
}

/// Запускает воркера в отдельном потоке, возвращает mpsc::Sender для общения с ним.
pub fn spawn_loader_thread(
    render_message_sender: mpsc::Sender<RenderMessage>,
    event_loop_proxy: winit::event_loop::EventLoopProxy<UserEvent>,
) -> Result<mpsc::Sender<LoaderMessage>> {
    let (loader_message_sender, loader_message_reciever) = mpsc::channel();
    let mut loader_thread = LoaderThread {
        render_message_sender,
        loader_message_reciever,
        event_loop_proxy,
        message_index: 0,
    };
    thread::spawn(move || loader_thread.run());
    Ok(loader_message_sender)
}
