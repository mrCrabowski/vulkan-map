#![allow(unsafe_op_in_unsafe_fn)]

use anyhow::{Result, anyhow};
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;

use vulkanalia::vk::ExtDebugUtilsExtensionInstanceCommands;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;

use winit::window::Window;

use crate::profile_span;

use cgmath::{Deg, SquareMatrix, point3, vec3};
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;
type Mat4 = cgmath::Matrix4<f32>;

use std::time::Instant;

use super::vulkan_app_data::AppData;
use super::vulkan_app_data::INDICES;
use super::vulkan_app_data::MAX_FRAMES_IN_FLIGHT;
use super::vulkan_app_data::UniformBufferObject;
use super::vulkan_app_data::VALIDATION_ENABLED;
use super::vulkan_app_data::create_command_buffers;
use super::vulkan_app_data::create_command_pool;
use super::vulkan_app_data::create_descriptor_pool;
use super::vulkan_app_data::create_descriptor_set_layout;
use super::vulkan_app_data::create_descriptor_sets;
use super::vulkan_app_data::create_framebuffers;
use super::vulkan_app_data::create_index_buffer;
use super::vulkan_app_data::create_instance;
use super::vulkan_app_data::create_logical_device;
use super::vulkan_app_data::create_pipeline;
use super::vulkan_app_data::create_render_pass;
use super::vulkan_app_data::create_swapchain;
use super::vulkan_app_data::create_swapchain_image_views;
use super::vulkan_app_data::create_sync_objects;
use super::vulkan_app_data::create_texture_image;
use super::vulkan_app_data::create_texture_image_view;
use super::vulkan_app_data::create_texture_sampler;
use super::vulkan_app_data::create_uniform_buffers;
use super::vulkan_app_data::create_vertex_buffer;
use super::vulkan_app_data::pick_physical_device;

/// Vulkan app.
#[derive(Clone, Debug)]
pub struct App {
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
    pub resized: bool,
    start: Instant,
}

impl App {
    pub unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&device, &mut data)?;
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        create_texture_image(&instance, &device, &mut data)?;
        create_texture_image_view(&device, &mut data)?;
        create_texture_sampler(&device, &mut data)?;
        create_vertex_buffer(&instance, &device, &mut data)?;
        create_index_buffer(&instance, &device, &mut data)?;
        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;
        Ok(Self {
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
        })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let _span = profile_span!("render");
        let frame = self.frame;
        let in_flight_fence = self.data.in_flight_fences[frame];

        // 1. Ждём завершения предыдущего frame
        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        // 2. Acquire image (сигналит image_available_semaphores[frame])
        let (image_index, _) = match self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[frame],
            vk::Fence::null(),
        ) {
            Ok(v) => v,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_index = image_index as usize;

        // 3. Если image уже используется — ждём ЕГО fence
        let image_fence = self.data.images_in_flight[image_index];
        if image_fence != vk::Fence::null() {
            self.device
                .wait_for_fences(&[image_fence], true, u64::MAX)?;
        }

        self.device.reset_fences(&[in_flight_fence])?;

        // Помечаем image как используемое этим frame
        self.data.images_in_flight[image_index] = in_flight_fence;

        // Обновление команд и данных
        self.update_command_buffer(image_index)?;
        self.update_uniform_buffer(image_index)?;

        // 3. Graphics queue (ждёт image_available_semaphores[frame], сигналит render_finished_semaphores[image_index])
        let render_finished = self.data.render_finished_semaphores[image_index as usize];
        let wait_semaphores = &[self.data.image_available_semaphores[frame]];
        let command_buffers = &[self.data.command_buffers[image_index as usize]];
        let signal_semaphores = &[render_finished];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        // 4. Present queue (ждёт render_finished_semaphores[image_index])
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);

        if result == Err(vk::ErrorCode::OUT_OF_DATE_KHR)
            || result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || self.resized
        {
            self.resized = false;
            self.recreate_swapchain(window)?;
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let time = self.start.elapsed().as_secs_f32();
        // let model = Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(90.0) * time);
        let model = Mat4::identity();

        let amplitude = 10.0;
        let z = 1.0 + amplitude * (1.0 + (time * 2.0 * std::f32::consts::PI / 10.0).sin());
        let view = Mat4::look_at_rh(
            point3(0.0, 1.0, z),
            point3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0),
        );
        let mut proj = cgmath::perspective(
            Deg(45.0),
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
            0.1,
            50.0,
        );
        proj[1][1] *= -1.0;

        let ubo = UniformBufferObject { model, view, proj };

        let memory = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.device
            .unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        let command_buffer = self.data.command_buffers[image_index];

        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        let info = vk::CommandBufferBeginInfo::builder();

        self.device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let clear_values = &[color_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        self.device
            .cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::INLINE);
        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline,
        );
        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.vertex_buffer], &[0]);
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.data.index_buffer,
            0,
            vk::IndexType::UINT16,
        );
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );
        self.device
            .cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);
        self.device.cmd_end_render_pass(command_buffer);
        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        let _span = profile_span!("recreate_swapchain");
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.device, &mut self.data)?;
        create_pipeline(&self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device
            .destroy_image_view(self.data.texture_image_view, None);
        self.device.destroy_image(self.data.texture_image, None);
        self.device
            .free_memory(self.data.texture_image_memory, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device
            .free_memory(self.data.vertex_buffer_memory, None);

        self.device
            .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);

        self.data
            .in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));
        self.data
            .render_finished_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data
            .image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.device
            .destroy_command_pool(self.data.command_pool, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.device
            .destroy_descriptor_pool(self.data.descriptor_pool, None);
        self.data
            .uniform_buffers
            .iter()
            .for_each(|b| self.device.destroy_buffer(*b, None));
        self.data
            .uniform_buffers_memory
            .iter()
            .for_each(|m| self.device.free_memory(*m, None));

        self.device
            .free_command_buffers(self.data.command_pool, &self.data.command_buffers);
        self.data
            .framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}
