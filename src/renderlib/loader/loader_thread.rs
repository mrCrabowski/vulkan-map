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
use std::fs::File;

struct LoaderThread {
    render_message_sender: mpsc::Sender<RenderMessage>,
    loader_message_reciever: mpsc::Receiver<LoaderMessage>,

    // TODO: Хардкод на winit для дерганья poll_messages и перерисовки в ивент лупе, нужна адекватная замена.
    event_loop_proxy: winit::event_loop::EventLoopProxy<UserEvent>,
}

impl LoaderThread {
    fn run(self) {
        while let Ok(cmd) = self.loader_message_reciever.recv() {
            match cmd {
                LoaderMessage::LoadImage => {
                    match self.load_image() {
                        Ok(raw) => {
                            self.render_message_sender
                                .send(RenderMessage::RawImage(raw))
                                .expect("Failed to send image");

                            self.event_loop_proxy
                                .send_event(UserEvent::RequestRenderPoll)
                                .ok();
                        }
                        Err(err) => error!("Failed to load image: {}", err),
                    };
                }
                LoaderMessage::Stop => break,
            }
        }
    }

    fn load_image(&self) -> Result<RawImage> {
        let _span = profile_span!("Loader::load_image");

        // TODO: REMOVE HARDCODE
        let image = File::open("C:/RustProjects/vulkan-map/resources/texture.png")
            .map_err(|b| anyhow!("Failed to load texture: {}", b))?;

        let decoder = png::Decoder::new(image);
        let mut reader = decoder.read_info()?;

        let mut pixels = vec![0; reader.info().raw_bytes()];
        reader.next_frame(&mut pixels)?;

        let size_bytes = reader.info().raw_bytes() as u32;
        let (width, height) = reader.info().size();
        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        let raw = RawImage {
            width,
            height,
            format: vk::Format::R8G8B8A8_SRGB,
            pixels,
            size_bytes,
            mip_levels,
        };
        Ok(raw)
    }
}

/// Запускает воркера в отдельном потоке, возвращает mpsc::Sender для общения с ним.
pub fn spawn_loader_thread(
    render_message_sender: mpsc::Sender<RenderMessage>,
    event_loop_proxy: winit::event_loop::EventLoopProxy<UserEvent>,
) -> Result<mpsc::Sender<LoaderMessage>> {
    let (loader_message_sender, loader_message_reciever) = mpsc::channel();
    let loader_thread = LoaderThread {
        render_message_sender,
        loader_message_reciever,
        event_loop_proxy,
    };
    thread::spawn(|| loader_thread.run());
    Ok(loader_message_sender)
}
