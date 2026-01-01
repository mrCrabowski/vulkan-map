use anyhow::Result;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoopBuilder;
use winit::window::WindowBuilder;

use log::*;

use std::env;

use vulkan_map::profile_span;
use vulkan_map::profiling::TracyPlotFPS;
use vulkan_map::renderlib::vulkan_render::App;

use vulkan_map::renderlib::user_event::UserEvent;

fn main() -> Result<()> {
    if env::var("RUST_LOG").is_err() {
        unsafe { env::set_var("RUST_LOG", "info") };
    }
    pretty_env_logger::init();

    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build()?;
    let event_loop_proxy = event_loop.create_proxy();

    let window = WindowBuilder::new()
        .with_title("Vulkan map")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let mut app = unsafe {
        let _span = profile_span!("App initialization");
        App::create(&window, event_loop_proxy)?
    };
    let mut fps_plot = TracyPlotFPS::create();
    let mut minimized = false;
    let mut needs_redraw = true; // Флаг, нужно ли рисовать кадр
    event_loop.run(move |event, elwt| match event {
        Event::UserEvent(UserEvent::RequestRedraw) => {
            debug!("Redraw requested");
            needs_redraw = true; // сцена изменилась
            window.request_redraw();
        }
        Event::UserEvent(UserEvent::RequestRenderPoll) => {
            let _span = profile_span!("Poll render events");
            app.poll_messages();
        }
        Event::AboutToWait => {
            // TODO: Убрать постоянную перерисовку, переделать вызов отрисовки только через proxy.send_event(UserEvent::RequestRedraw).ok();
            needs_redraw = true;
            window.request_redraw();
        }
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                if needs_redraw {
                    fps_plot.on_redraw();
                    unsafe {
                        app.render(&window).unwrap();
                    }
                    needs_redraw = false;
                }
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                    needs_redraw = true;
                    window.request_redraw();
                }
            }
            WindowEvent::CloseRequested => {
                let _span = profile_span!("App closing");
                elwt.exit();
                unsafe {
                    app.destroy();
                }
            }
            _ => {}
        },
        _ => {}
    })?;

    Ok(())
}
