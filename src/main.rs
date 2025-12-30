use anyhow::Result;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use std::env;

use vulkan_map::profile_span;
use vulkan_map::profiling::TracyPlotFPS;
use vulkan_map::renderlib::vulkan_render::App;

fn main() -> Result<()> {
    if env::var("RUST_LOG").is_err() {
        unsafe { env::set_var("RUST_LOG", "info") };
    }
    pretty_env_logger::init();

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Vulkan map")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let mut app = unsafe {
        let _span = profile_span!("App initialization");
        App::create(&window)?
    };
    let mut minimized = false;
    let mut fps_plot = TracyPlotFPS::create();
    event_loop.run(move |event, elwt| match event {
        Event::AboutToWait => window.request_redraw(),
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                fps_plot.on_redraw();

                unsafe {
                    app.render(&window).unwrap();
                }
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
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
