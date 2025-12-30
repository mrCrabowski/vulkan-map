#[cfg(feature = "tracy")]
use std::time::{Duration, Instant};
#[cfg(feature = "tracy")]
use tracy_client::plot;
#[cfg(feature = "tracy")]
pub use tracy_client::span;

/// Guard для профилировки. Используется через `let _span = profile_span!("name");`
#[must_use = "ProfileSpan must be bound to a variable to keep the profiling scope alive"]
pub struct ProfileSpan(
    #[cfg(feature = "tracy")]
    #[allow(dead_code)]
    tracy_client::Span,
);

#[cfg(feature = "tracy")]
impl ProfileSpan {
    pub fn new(span: tracy_client::Span) -> Self {
        Self(span)
    }
}

#[cfg(not(feature = "tracy"))]
impl ProfileSpan {
    pub fn new() -> Self {
        Self()
    }
}

/// Макрос, который возвращает guard, аналогично tracy_client::span!
/// Использование:
/// ```rust
/// let _span = profile_span!("Render");
/// ```
#[macro_export]
macro_rules! profile_span {
    ($name:literal) => {{
        #[cfg(feature = "tracy")]
        {
            let span = tracy_client::span!($name);
            $crate::profiling::ProfileSpan::new(span)
        }

        #[cfg(not(feature = "tracy"))]
        {
            $crate::profiling::ProfileSpan::new()
        }
    }};
}

#[cfg(not(feature = "tracy"))]
/// Заглушка TracyPlotFPS
pub struct TracyPlotFPS;

#[cfg(not(feature = "tracy"))]
impl TracyPlotFPS {
    pub fn create() -> Self {
        Self
    }
    pub fn on_redraw(&mut self) {}
}

#[cfg(feature = "tracy")]
/// Класс для отрисовки графика FPS через Tracy
pub struct TracyPlotFPS {
    frame_count: u32,
    last_fps_time: Instant,
    fps: u32,
}

#[cfg(feature = "tracy")]
impl TracyPlotFPS {
    pub fn create() -> Self {
        let frame_count: u32 = 0;
        let last_fps_time = Instant::now();
        let fps: u32 = 0;
        Self {
            frame_count,
            last_fps_time,
            fps,
        }
    }

    pub fn on_redraw(&mut self) {
        self.frame_count += 1;

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_time);

        if elapsed >= Duration::from_secs(1) {
            self.fps = self.frame_count;
            self.frame_count = 0;
            self.last_fps_time = now;
            plot!("FPS", self.fps as f64);
        }
    }
}
