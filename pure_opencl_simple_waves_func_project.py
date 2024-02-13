import pygame
import pyopencl as cl
import numpy as np

# Initialize Pygame
pygame.init()

# Set up Pygame window
window_dim = 512
screen = pygame.display.set_mode((window_dim, window_dim))
pygame.display.set_caption("OpenCL Rendering")

# Create OpenCL context and queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Check the device type
device_type = device.type
if device_type == cl.device_type.GPU:
    print("The kernel is running on a GPU.")
elif device_type == cl.device_type.CPU:
    print("The kernel is running on a CPU.")
else:
    print("The device type is unknown or not recognized as GPU or CPU.")

# Create buffers
screen_np = np.zeros((window_dim, window_dim, 3), dtype=np.uint8)
screen_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=screen_np)

# Create initial positions buffer
"""
wave_instances = 128
inst_pos_np = np.array(([100, 100],
                        [150, 100],
                        [200, 100],
                        [250, 100],
                        [300, 100],
                        [350, 100],
                        [400, 100],
                        [450, 100]), dtype = np.uint32)
"""
wave_instances = 128
inst_pos_np = np.random.randint(16, window_dim - 17, size = (wave_instances, 2), dtype=np.uint32)
inst_pos_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=inst_pos_np)

squares_kernel = """
    __kernel void func(__global uchar *screen, __global uint2 *inst_pos, uint window_dim, uint wave_instances, float time) {
        uint xi = get_global_id(0);
        uint yi = get_global_id(1);
        uint index = 3 * (xi + yi * window_dim);
        uint red = 0;
        uint green = 0;
        uint blue = 0;

        const uint max_wave_instances = 128;
        local uint2 local_inst_pos[max_wave_instances];

        for (int i = 0; i < wave_instances; ++i)
            local_inst_pos[i] = inst_pos[i];

        for(uint wave = 0; wave < wave_instances; wave ++) {
            float dx = local_inst_pos[wave].x - (float)xi;
            float dy = local_inst_pos[wave].y - (float)yi;
            float radius = sqrt(dx * dx + dy * dy);

            float sigma = 200.0f; // Adjust this parameter to control the width of the bell curve
            float bellCurve = exp(-(radius * radius) / (2.0f * sigma * sigma));

            int z = (int)((sin(radius * 0.1 - time) + 1.f) * (127.f / 16.f) * bellCurve);
            red += z;
            green += z;
            blue += z;
        }

        screen[index] = red;
        screen[index + 1] = green;
        screen[index + 2] = blue;
    }
"""

# Compile the programs
program = cl.Program(context, squares_kernel).build()

# Set up clock for tracking FPS
clock = pygame.time.Clock()

timer = 0.

# Main Pygame loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if not(mouse_x > 16 and mouse_x < window_dim - 17 and mouse_y > 16 and mouse_y < window_dim - 17):
        mouse_x, mouse_y = window_dim * 0.5, window_dim * 0.5
    # Update the position buffer
    inst_pos_np[0, 1] = np.uint32(mouse_x)
    inst_pos_np[0, 0] = np.uint32(mouse_y)
    cl.enqueue_copy(queue, inst_pos_buf, inst_pos_np).wait()

    # Execute the kernel
    global_size = (window_dim, window_dim)
    local_size = None
    program.func(queue, global_size, local_size, screen_buf, inst_pos_buf, np.uint32(window_dim), np.uint32(wave_instances), np.float32(timer)).wait()
    timer += 0.1 * 3.1415
    if timer > 2 * 3.1415:
        timer = 0.
    # Read back the result
    cl.enqueue_copy(queue, screen_np, screen_buf).wait()

    # Convert to Pygame surface
    screen_surface = pygame.surfarray.make_surface(screen_np)

    # Blit the surface onto the Pygame window
    screen.blit(screen_surface, (0, 0))

    # Update the Pygame window
    pygame.display.flip()

    # Cap the frame rate to 60 FPS
    clock.tick(60)

    # Calculate FPS
    fps = clock.get_fps()

    # Print FPS to the console
    print(f"FPS: {fps:.2f}")

pygame.quit()