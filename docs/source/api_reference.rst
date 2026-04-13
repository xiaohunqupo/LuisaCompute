API Reference
=============

This document provides a comprehensive reference for the LuisaCompute C++ API.
For introductory material, see :doc:`getting_started`; for detailed DSL usage, see :doc:`dsl`;
for runtime resource management, see :doc:`resources`.

.. contents:: Table of Contents
   :depth: 3
   :local:


Core Types
----------

LuisaCompute provides GLSL/HLSL-style vector and matrix types for graphics computing.
All types live in the ``luisa`` namespace and are defined in ``<luisa/core/basic_types.h>``.

Scalar Types
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Type
     - Size
     - Description
   * - ``bool``
     - 1 byte
     - Boolean
   * - ``int`` / ``int32_t``
     - 4 bytes
     - Signed 32-bit integer
   * - ``uint`` / ``uint32_t``
     - 4 bytes
     - Unsigned 32-bit integer
   * - ``float``
     - 4 bytes
     - IEEE 754 single-precision floating point
   * - ``half``
     - 2 bytes
     - IEEE 754 half-precision floating point
   * - ``short`` / ``int16_t``
     - 2 bytes
     - Signed 16-bit integer
   * - ``ushort`` / ``uint16_t``
     - 2 bytes
     - Unsigned 16-bit integer
   * - ``slong`` / ``int64_t``
     - 8 bytes
     - Signed 64-bit integer
   * - ``ulong`` / ``uint64_t``
     - 8 bytes
     - Unsigned 64-bit integer
   * - ``byte`` / ``int8_t``
     - 1 byte
     - Signed 8-bit integer
   * - ``ubyte`` / ``uint8_t``
     - 1 byte
     - Unsigned 8-bit integer

Vector Types
^^^^^^^^^^^^

Vector types follow the naming convention ``<scalar><N>`` where N is 2, 3, or 4:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Boolean
     - Integer
     - Unsigned Integer
     - Floating Point
   * - ``bool2``
     - ``int2``
     - ``uint2``
     - ``float2``
   * - ``bool3``
     - ``int3``
     - ``uint3``
     - ``float3``
   * - ``bool4``
     - ``int4``
     - ``uint4``
     - ``float4``

Additional vector types: ``half2/3/4``, ``short2/3/4``, ``ushort2/3/4``,
``byte2/3/4``, ``ubyte2/3/4``, ``slong2/3/4``, ``ulong2/3/4``, ``double2/3/4``.

.. warning::

   3-component vectors and matrices are **aligned to 16 bytes** (same as 4-component).
   This matches GPU alignment requirements but differs from packed C++ structs.
   64-bit vector/matrix types are generally **not supported on GPUs**.

**Construction:**

.. code-block:: cpp

   // Scalar broadcast
   float3 v = make_float3(1.0f);            // (1.0, 1.0, 1.0)

   // Component-wise
   float3 v = make_float3(1.0f, 2.0f, 3.0f);

   // From smaller vector + scalar
   float3 v = make_float3(make_float2(1.0f, 2.0f), 3.0f);

   // Type conversion
   float3 v = make_float3(make_int3(1, 2, 3));  // (1.0, 2.0, 3.0)

   // Factory methods
   auto z = float3::zero();                  // (0, 0, 0)
   auto o = float3::one();                   // (1, 1, 1)

**Element access:**

.. code-block:: cpp

   float3 v(1.0f, 2.0f, 3.0f);
   float x = v.x;     // Named: .x, .y, .z, .w
   float y = v[1];    // Indexed

**Operators** — arithmetic (``+``, ``-``, ``*``, ``/``), comparison (``==``, ``<``, etc.),
bitwise (``~``, ``&``, ``|``, ``^``, ``<<``, ``>>``), and boolean logic (``||``, ``&&``)
all work component-wise. Scalar-vector operations broadcast the scalar.

**Boolean reductions:**

.. code-block:: cpp

   bool2 b(true, false);
   bool r = any(b);    // true
   bool r = all(b);    // false
   bool r = none(b);   // false

Matrix Types
^^^^^^^^^^^^

Column-major square matrices:

- ``float2x2``, ``float3x3``, ``float4x4``
- ``double2x2``, ``double3x3``, ``double4x4``
- ``half2x2``, ``half3x3``, ``half4x4``

.. code-block:: cpp

   // Identity matrix
   float4x4 m = make_float4x4(1.0f);

   // Diagonal matrix
   float3x3 m = make_float3x3(2.0f);           // 2 * identity

   // From column vectors
   float2x2 m = make_float2x2(
       make_float2(1.0f, 0.0f),    // Column 0
       make_float2(0.0f, 1.0f));   // Column 1

   // Element-wise (row-major order in arguments)
   float2x2 m = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);

   // Column access
   float3 col = m[0];    // First column

   // Matrix-vector multiply
   float3 result = m * v; // v is treated as column vector

Math Functions
^^^^^^^^^^^^^^

Defined in ``<luisa/core/mathematics.h>``. All functions work on both scalars and vectors (component-wise).

**Trigonometric** (radians):
``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``atan2``, ``sinh``, ``cosh``

**Exponential:**
``pow``, ``exp``, ``exp2``, ``log``, ``log2``, ``sqrt``

**Common:**
``abs``, ``sign``, ``floor``, ``ceil``, ``round``, ``trunc``, ``fract``, ``fmod``

**Clamping and interpolation:**
``min``, ``max``, ``clamp``, ``lerp``, ``select``, ``fma``

**Vector operations:**
``dot``, ``cross`` (3D only), ``length``, ``normalize``, ``distance``

**Matrix operations:**
``transpose``, ``inverse``, ``determinant``

**Transformations** (return ``float4x4``):
``translation(x, y, z)``, ``scaling(x, y, z)``, ``scaling(uniform)``,
``rotation(axis, angle)``

**Angle conversion:**
``radians(degrees)``, ``degrees(radians)``

**Constants** (in ``luisa::constants``):
``pi``, ``pi_over_2``, ``pi_over_4``, ``two_pi``, ``inv_pi``, ``e``

.. code-block:: cpp

   #include <luisa/core/mathematics.h>
   using namespace luisa;

   float3 a = make_float3(1.0f, 0.0f, 0.0f);
   float3 b = make_float3(0.0f, 1.0f, 0.0f);

   float  d   = dot(a, b);        // 0.0
   float3 c   = cross(a, b);      // (0, 0, 1)
   float  len = length(a);        // 1.0
   float3 n   = normalize(a + b); // (0.707, 0.707, 0)

   float4x4 T = translation(1.0f, 2.0f, 3.0f);
   float4x4 R = rotation(make_float3(0, 0, 1), constants::pi / 4.0f);

Type Traits
^^^^^^^^^^^

Defined in ``<luisa/core/basic_traits.h>``. Compile-time predicates for type introspection:

.. code-block:: cpp

   luisa::is_scalar_v<T>          // bool, int, uint, float, half, ...
   luisa::is_vector_v<T>          // Any vector type
   luisa::is_vector_v<T, N>       // Vector with N components
   luisa::is_matrix_v<T>          // Any matrix type
   luisa::is_basic_v<T>           // Scalar, vector, or matrix
   luisa::vector_element_t<T>     // Element type of a vector
   luisa::vector_dimension_v<T>   // Number of components (1 for scalars)


Runtime
-------

The runtime layer provides a unified API for creating and managing GPU resources,
compiling shaders, and scheduling commands across different backends.

All runtime types live in ``luisa::compute``.

Context
^^^^^^^

.. doxygenclass:: luisa::compute::Context
   :members:

The ``Context`` is the entry point for the entire runtime. It manages backend plugin
discovery and device creation.

.. code-block:: cpp

   #include <luisa/runtime/context.h>

   Context context{argv[0]};                   // From executable path
   Context context{argv[0], "/data/dir"};      // With custom data directory

   // Query installed backends
   for (auto &backend : context.installed_backends()) {
       auto devices = context.backend_device_names(backend);
   }

   // Create devices
   Device device = context.create_device("cuda");
   Device device = context.create_default_device();  // First available

Device
^^^^^^

.. doxygenclass:: luisa::compute::Device
   :members:

A ``Device`` represents a specific GPU or CPU backend. It is the factory for all resources,
streams, events, and shader compilation.

.. code-block:: cpp

   #include <luisa/runtime/device.h>

   Device device = context.create_device("cuda");

   // With configuration
   DeviceConfig config{
       .device_index = 0,
       .inqueue_buffer_limit = false
   };
   Device device = context.create_device("cuda", &config, true /* validation */);

   // Query properties
   auto name       = device.backend_name();      // "cuda"
   auto warp_size  = device.compute_warp_size();  // e.g. 32
   auto granularity = device.memory_granularity();

**Resource creation methods** (summary):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Description
   * - ``create_buffer<T>(n)``
     - Linear buffer of ``n`` elements of type ``T``
   * - ``create_byte_buffer(size)``
     - Untyped byte buffer
   * - ``create_image<T>(storage, w, h, mips, simultaneous)``
     - 2D texture
   * - ``create_volume<T>(storage, w, h, d, mips)``
     - 3D texture
   * - ``create_bindless_array(slots)``
     - Bindless resource array
   * - ``create_stream(tag)``
     - Command queue
   * - ``create_event()``
     - Binary synchronization event
   * - ``create_timeline_event()``
     - Timeline-based synchronization
   * - ``create_swapchain(stream, options)``
     - Window presentation surface
   * - ``create_mesh(vertices, triangles)``
     - Triangle mesh for ray tracing
   * - ``create_accel(option)``
     - Top-level acceleration structure
   * - ``create_indirect_dispatch_buffer(cap)``
     - Indirect dispatch argument buffer
   * - ``create_depth_buffer(format, size)``
     - Depth buffer for rasterization
   * - ``compile(kernel)``
     - Compile kernel into a ``Shader``

Stream
^^^^^^

.. doxygenclass:: luisa::compute::Stream
   :members:

A ``Stream`` is an asynchronous command queue. Commands are submitted with ``operator<<``
and execute in submission order. Multiple streams run concurrently.

.. code-block:: cpp

   #include <luisa/runtime/stream.h>

   Stream stream = device.create_stream();                      // Default: COMPUTE
   Stream compute  = device.create_stream(StreamTag::COMPUTE);  // Compute workloads
   Stream graphics = device.create_stream(StreamTag::GRAPHICS); // Rasterization + present
   Stream copy     = device.create_stream(StreamTag::COPY);     // Data transfers

   stream.set_name("my compute stream");   // Debug label

   // Submit commands
   stream << buffer.copy_from(host_data)
          << shader(args...).dispatch(n)
          << synchronize();

   // Host callbacks
   stream << shader(args...).dispatch(n)
          << [&]() { std::cout << "Done!" << std::endl; };

Event and TimelineEvent
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: luisa::compute::Event
   :members:

.. doxygenclass:: luisa::compute::TimelineEvent
   :members:

**Event** provides binary signal/wait synchronization between streams:

.. code-block:: cpp

   Event event = device.create_event();

   stream_a << compute_shader().dispatch(n)
            << event.signal();

   stream_b << event.wait()               // Blocks until signal
            << dependent_shader().dispatch(n);

   event.synchronize();                   // Block host until signaled

**TimelineEvent** supports monotonically increasing counter values, useful for frame pacing:

.. code-block:: cpp

   TimelineEvent timeline = device.create_timeline_event();
   uint64_t frame = 0;

   // Render loop with triple buffering
   while (running) {
       if (frame >= 3) timeline.synchronize(frame - 2);

       stream << render_shader().dispatch(w, h)
              << timeline.signal(++frame);
   }

CommandList
^^^^^^^^^^^

Commands can be grouped into explicit command lists for batch submission:

.. code-block:: cpp

   #include <luisa/runtime/command_list.h>

   CommandList cmdlist = CommandList::create();
   cmdlist << shader.dispatch(w, h)
           << buffer.copy_to(host_data);

   stream << cmdlist.commit() << synchronize();

The runtime automatically analyzes dependencies between commands in a list and
reorders them to maximize GPU utilization.


Resources
---------

Buffer<T>
^^^^^^^^^

.. doxygenclass:: luisa::compute::Buffer
   :members:

Typed linear GPU memory for structured data. Elements must be scalar, vector, matrix,
array, or ``LUISA_STRUCT``-reflected types.

.. code-block:: cpp

   #include <luisa/runtime/buffer.h>

   Buffer<float> buf = device.create_buffer<float>(1024);
   buf.set_name("my buffer");          // Debug label

   // Host-device transfers
   stream << buf.copy_from(host_ptr)   // Upload
          << buf.copy_to(host_ptr);    // Download

   // Buffer-to-buffer copy
   stream << dst.copy_from(src);

   // Sub-buffer views
   BufferView<float> sub = buf.view(100, 500);  // Offset 100, count 500
   stream << sub.copy_from(host_ptr);

**In kernels** (via ``BufferVar<T>`` or aliases like ``BufferFloat``):

.. code-block:: cpp

   Kernel1D kernel = [&](BufferFloat buf) noexcept {
       auto idx = dispatch_id().x;
       Float val = buf.read(idx);
       buf.write(idx, val + 1.0f);

       // Atomic operations
       buf.atomic(0u).fetch_add(1.0f);

       // Volatile (coherent) access
       Float v = buf.volatile_read(idx);
       buf.volatile_write(idx, v);
   };

Image<T>
^^^^^^^^

.. doxygenclass:: luisa::compute::Image
   :members:

2D texture with hardware-accelerated caching and format conversion.
The template parameter ``T`` (``float``, ``int``, or ``uint``) determines how pixel values
are interpreted when reading/writing — the actual storage format is specified by ``PixelStorage``.

.. code-block:: cpp

   #include <luisa/runtime/image.h>

   // RGBA8 image with float read/write (automatic byte ↔ float conversion)
   Image<float> img = device.create_image<float>(PixelStorage::BYTE4, 1024, 1024);

   // HDR image
   Image<float> hdr = device.create_image<float>(PixelStorage::FLOAT4, 1920, 1080);

   // With mipmaps
   Image<float> mip = device.create_image<float>(PixelStorage::BYTE4, 1024, 1024, 10);

   // Simultaneous access (for multi-stream read/write)
   Image<float> shared = device.create_image<float>(PixelStorage::BYTE4, w, h, 1, true);

   // Host transfer
   stream << img.copy_from(pixels_ptr) << img.copy_to(pixels_ptr);

   // Mipmap views
   ImageView<float> level0 = img.view(0);

**In kernels** (via ``ImageFloat``, ``ImageInt``, ``ImageUInt``):

.. code-block:: cpp

   Kernel2D kernel = [&](ImageFloat image) noexcept {
       UInt2 coord = dispatch_id().xy();
       Float4 color = image.read(coord);
       image.write(coord, color * 2.0f);
   };

Volume<T>
^^^^^^^^^

.. doxygenclass:: luisa::compute::Volume
   :members:

3D texture. API mirrors ``Image<T>`` with an additional depth dimension.

.. code-block:: cpp

   #include <luisa/runtime/volume.h>

   Volume<float> vol = device.create_volume<float>(PixelStorage::FLOAT4, 256, 256, 256);

   // In kernel
   Kernel3D kernel = [&](VolumeFloat vol) noexcept {
       UInt3 coord = dispatch_id().xyz();
       Float4 val = vol.read(coord);
       vol.write(coord, val * 2.0f);
   };

ByteBuffer
^^^^^^^^^^

Untyped (raw byte) GPU buffer for manual memory layout:

.. code-block:: cpp

   #include <luisa/runtime/byte_buffer.h>

   ByteBuffer raw = device.create_byte_buffer(4096);

   // In kernel — templated read/write at byte offsets
   Kernel1D kernel = [&](Var<ByteBuffer> bb) noexcept {
       Float3 v = bb.read<float3>(0u);     // Read float3 at byte offset 0
       bb.write(16u, v + 1.0f);            // Write at byte offset 16

       // Volatile access
       Float3 v2 = bb.volatile_read<float3>(32u);
       bb.volatile_write(48u, v2);
   };

IndirectDispatchBuffer
^^^^^^^^^^^^^^^^^^^^^^

Buffer of indirect dispatch arguments for GPU-driven dispatch:

.. code-block:: cpp

   #include <luisa/runtime/dispatch_buffer.h>

   IndirectDispatchBuffer indirect = device.create_indirect_dispatch_buffer(64);

   // Populate in a kernel, then dispatch another kernel indirectly
   stream << populate_shader(indirect).dispatch(1u)
          << work_shader(args...).dispatch(indirect);

Swapchain
^^^^^^^^^

Presents rendered images to a window surface. Requires a ``StreamTag::GRAPHICS`` stream.

.. code-block:: cpp

   #include <luisa/runtime/swapchain.h>

   Swapchain swapchain = device.create_swapchain(
       stream,
       SwapchainOption{
           .display = window.native_display(),
           .window  = window.native_handle(),
           .size    = make_uint2(1920, 1080),
           .wants_hdr = false,
           .wants_vsync = true,
           .back_buffer_count = 3
       });

   // Query the backend's native pixel storage (for matching Image format)
   PixelStorage storage = swapchain.backend_storage();
   Image<float> framebuffer = device.create_image<float>(storage, 1920, 1080);

   // Present in render loop
   stream << render_shader(framebuffer).dispatch(1920, 1080)
          << swapchain.present(framebuffer.view(0));

BindlessArray
^^^^^^^^^^^^^

.. doxygenclass:: luisa::compute::BindlessArray
   :members:

Dynamic resource array for shader-side indexing without fixed binding slots.
Supports buffers, 2D textures, and 3D textures.

.. code-block:: cpp

   #include <luisa/runtime/bindless_array.h>

   BindlessArray heap = device.create_bindless_array(65536);

   // Bind resources
   heap.emplace_on_update(0, buffer);
   heap.emplace_on_update(1, image, Sampler::linear_linear_mirror());

   // Commit changes
   stream << heap.update() << synchronize();

   // In kernel
   Kernel1D kernel = [&](Var<BindlessArray> heap, UInt slot) noexcept {
       Float val = heap.buffer<float>(slot).read(0u);
       Float4 color = heap.texture2d(slot).sample(uv);
   };


Pixel Formats and Sampling
---------------------------

PixelStorage
^^^^^^^^^^^^

``PixelStorage`` specifies the internal memory layout of image and volume textures.
Defined in ``<luisa/runtime/rhi/pixel.h>``.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Storage
     - Channels
     - Bits/Channel
     - Notes
   * - ``BYTE1/2/4``
     - 1/2/4
     - 8
     - Unsigned normalized when ``T=float``, signed integer when ``T=int``
   * - ``SHORT1/2/4``
     - 1/2/4
     - 16
     - Unsigned normalized or unsigned integer
   * - ``INT1/2/4``
     - 1/2/4
     - 32
     - Full integer precision
   * - ``HALF1/2/4``
     - 1/2/4
     - 16
     - IEEE 754 half-precision float
   * - ``FLOAT1/2/4``
     - 1/2/4
     - 32
     - IEEE 754 single-precision float
   * - ``R10G10B10A2``
     - 4
     - 10/10/10/2
     - Packed HDR format
   * - ``R11G11B10``
     - 3
     - 11/11/10
     - Packed HDR float format
   * - ``BC1`` – ``BC7``
     - varies
     - block
     - Block-compressed formats (read-only in shaders)
   * - ``BYTE4_SRGB``
     - 4
     - 8
     - sRGB color space, automatic linear ↔ sRGB conversion

The template parameter ``T`` on ``Image<T>`` and ``Volume<T>`` selects the
read/write interpretation:

- ``Image<float>`` + ``BYTE4`` → automatic ``[0,255] ↔ [0.0, 1.0]`` conversion
- ``Image<int>``   + ``BYTE4`` → raw signed integer access
- ``Image<uint>``  + ``BYTE4`` → raw unsigned integer access

Sampler
^^^^^^^

``Sampler`` controls texture filtering and addressing when sampling in ``BindlessArray``.
Defined in ``<luisa/runtime/rhi/sampler.h>``.

**Filter modes:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Filter
     - Description
   * - ``POINT``
     - Nearest-neighbor (no filtering)
   * - ``LINEAR_POINT``
     - Bilinear within a mip level, nearest between mip levels
   * - ``LINEAR_LINEAR``
     - Trilinear (bilinear + linear mip interpolation)
   * - ``ANISOTROPIC``
     - Anisotropic filtering

**Address modes:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Address
     - Description
   * - ``EDGE``
     - Clamp to edge texel
   * - ``REPEAT``
     - Tile (wrap around)
   * - ``MIRROR``
     - Mirror at boundaries
   * - ``ZERO``
     - Return zero outside [0, 1]

**Factory methods** — convenience combinators:

.. code-block:: cpp

   Sampler::point_edge()
   Sampler::linear_linear_mirror()
   Sampler::anisotropic_repeat()
   // ... all 16 combinations: {filter}_{address}()

StreamTag
^^^^^^^^^

Selects the type of work a ``Stream`` can execute.
Defined in ``<luisa/runtime/rhi/stream_tag.h>``.

- ``StreamTag::COMPUTE`` — compute shader dispatch (default)
- ``StreamTag::GRAPHICS`` — rasterization and window presentation
- ``StreamTag::COPY`` — data transfer (may overlap with compute/graphics)

DepthFormat
^^^^^^^^^^^

Depth buffer precision, used with ``device.create_depth_buffer()``:

- ``DepthFormat::D16`` — 16-bit depth
- ``DepthFormat::D24S8`` — 24-bit depth + 8-bit stencil
- ``DepthFormat::D32`` — 32-bit floating-point depth


Shader Compilation
------------------

Kernels are compiled into ``Shader`` objects by the device. A shader is a compiled,
dispatchable unit of GPU work.

.. code-block:: cpp

   // Define a 2D kernel
   Kernel2D fill = [&](ImageFloat image) noexcept {
       UInt2 coord = dispatch_id().xy();
       image.write(coord, make_float4(1.0f, 0.0f, 0.0f, 1.0f));
   };

   // Compile
   auto shader = device.compile(fill);

   // Dispatch
   stream << shader(image.view(0)).dispatch(1024, 1024);

The ``Shader<N, Args...>`` template is parameterized by dispatch dimension (1, 2, or 3)
and the kernel's parameter types. The ``device.compile()`` call blocks the calling thread;
for large kernels this may take significant time. Consider compiling multiple kernels
concurrently (e.g., with a thread pool).

Most backends cache compiled shaders on disk at ``<build-folder>/bin/.cache``.


Ray Tracing
-----------

LuisaCompute provides hardware-accelerated ray tracing via a two-level acceleration
structure: **Mesh** (bottom-level) and **Accel** (top-level).

Mesh
^^^^

A triangle mesh for ray-scene intersection. Created from vertex and index buffers.

.. code-block:: cpp

   #include <luisa/runtime/rtx/mesh.h>

   Buffer<float3>   vertices  = device.create_buffer<float3>(num_verts);
   Buffer<Triangle> triangles = device.create_buffer<Triangle>(num_tris);
   // ... fill buffers ...

   Mesh mesh = device.create_mesh(vertices, triangles);
   stream << mesh.build();   // Build BVH

``Triangle`` is a struct of three ``uint`` indices into the vertex buffer.

Accel
^^^^^

.. doxygenclass:: luisa::compute::Accel
   :members:

Top-level acceleration structure containing mesh instances with transforms and visibility masks.

.. code-block:: cpp

   #include <luisa/runtime/rtx/accel.h>

   Accel accel = device.create_accel();

   // Add mesh instances
   accel.emplace_back(mesh, transform_matrix);
   accel.emplace_back(mesh, transform_matrix, visibility_mask);

   // Build
   stream << mesh.build() << accel.build();

   // Update transforms at runtime
   accel.set_transform_on_update(instance_index, new_transform);
   stream << accel.update_instance_buffer() << accel.build();

**In kernels** (via ``AccelVar``):

.. code-block:: cpp

   Kernel2D trace = [&](AccelVar accel, ImageFloat image) noexcept {
       // Create a ray
       Var<Ray> ray = make_ray(origin, direction);

       // Closest-hit intersection
       Var<SurfaceHit> hit = accel.intersect(ray, {});

       $if (!hit->miss()) {
           // hit.inst     — instance index
           // hit.prim     — triangle index within the mesh
           // hit.bary     — barycentric coordinates (float2)
           // hit.committed_ray_t — intersection distance
       };

       // Any-hit query (shadow rays, faster)
       Bool occluded = accel.intersect_any(ray, {});
   };

Ray and Hit Types
^^^^^^^^^^^^^^^^^

Defined in ``<luisa/runtime/rtx/ray.h>`` and ``<luisa/runtime/rtx/hit.h>``.

.. code-block:: cpp

   struct Ray {
       float3 origin;
       float  t_min;     // Minimum ray parameter (default 0)
       float3 direction;
       float  t_max;     // Maximum ray parameter (default infinity)
   };

   // In DSL
   Var<Ray> ray = make_ray(origin, direction);
   Var<Ray> ray = make_ray(origin, direction, t_min, t_max);

``TriangleHit`` fields (in DSL):

- ``hit.inst`` — instance index in the ``Accel``
- ``hit.prim`` — primitive (triangle) index in the ``Mesh``
- ``hit.bary`` — barycentric coordinates (``Float2``)
- ``hit.committed_ray_t`` — distance along the ray
- ``hit.miss()`` — returns ``Bool``, true if no intersection

Curve
^^^^^

Curve primitives for hair/fur rendering. Supported basis types:

- ``CurveBasis::CUBIC_BSPLINE``
- ``CurveBasis::CATMULL_ROM``
- ``CurveBasis::LINEAR``
- ``CurveBasis::BEZIER``

.. code-block:: cpp

   #include <luisa/runtime/rtx/curve.h>

   Curve curve = device.create_curve(
       CurveBasis::CUBIC_BSPLINE,
       control_point_buffer,    // Buffer<float4>: xyz = position, w = radius
       segment_buffer);         // Buffer<uint>: indices into control points

   stream << curve.build();
   accel.emplace_back(curve, transform);

ProceduralPrimitive
^^^^^^^^^^^^^^^^^^^

Custom intersection primitives defined by AABBs (axis-aligned bounding boxes):

.. code-block:: cpp

   #include <luisa/runtime/rtx/procedural_primitive.h>

   ProceduralPrimitive proc = device.create_procedural_primitive(aabb_buffer);
   stream << proc.build();
   accel.emplace_back(proc, transform);

Ray queries in kernels can test intersections against procedural primitives using
``RayQueryAll`` / ``RayQueryAny`` with custom intersection logic.


DSL Types
---------

The DSL types wrap C++ types for device-side (GPU) computation. They live in
``luisa::compute`` and are defined in ``<luisa/dsl/syntax.h>``.

Var<T> and Expr<T>
^^^^^^^^^^^^^^^^^^

``Var<T>`` is a mutable device variable. ``Expr<T>`` is a read-only reference
(analogous to ``const T&`` vs ``T``).

.. code-block:: cpp

   // Var<T> creates a new device variable
   Float x = 1.0f;           // Var<float>
   Float3 v = make_float3(1.0f, 2.0f, 3.0f);

   // Expr<T> references an expression without creating a variable
   Expr<float> sum = x + 1.0f;   // No variable allocation

   // def<T>() converts host values to DSL
   auto d = def(3.14f);          // Float
   auto v = def(host_float3);    // Float3

**Common aliases:**

.. list-table::
   :widths: 30 30 40

   * - ``Float``
     - ``Var<float>``
     -
   * - ``Int``
     - ``Var<int>``
     -
   * - ``UInt``
     - ``Var<uint>``
     -
   * - ``Bool``
     - ``Var<bool>``
     -
   * - ``Float2/3/4``
     - ``Var<float2/3/4>``
     -
   * - ``Int2/3/4``
     - ``Var<int2/3/4>``
     -
   * - ``UInt2/3/4``
     - ``Var<uint2/3/4>``
     -
   * - ``Float2x2/3x3/4x4``
     - ``Var<float2x2/3x3/4x4>``
     -

Resource DSL Proxies
^^^^^^^^^^^^^^^^^^^^

When passed to kernels/callables, runtime resources become DSL proxy types:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Runtime Type
     - DSL Parameter Type
     - Aliases
   * - ``Buffer<T>``
     - ``BufferVar<T>``
     - ``BufferFloat``, ``BufferInt``, ``BufferUInt``, ...
   * - ``Image<T>``
     - ``ImageVar<T>``
     - ``ImageFloat``, ``ImageInt``, ``ImageUInt``
   * - ``Volume<T>``
     - ``VolumeVar<T>``
     - ``VolumeFloat``, ``VolumeInt``, ``VolumeUInt``
   * - ``ByteBuffer``
     - ``Var<ByteBuffer>``
     -
   * - ``BindlessArray``
     - ``Var<BindlessArray>``
     -
   * - ``Accel``
     - ``AccelVar``
     -

Shared<T>
^^^^^^^^^

Fast on-chip shared memory accessible by all threads in a block.

.. code-block:: cpp

   Kernel1D kernel = [&]() noexcept {
       Shared<float> shared_data{256};     // 256 float elements

       shared_data[thread_x()] = some_value;
       sync_block();                       // Barrier
       Float neighbor = shared_data[thread_x() + 1u];

       // Atomic operations on shared memory
       shared_data.atomic(0u).fetch_add(1.0f);
   };

AtomicRef<T>
^^^^^^^^^^^^

Provides atomic operations on buffer or shared memory elements. Obtained via ``.atomic(index)``:

.. code-block:: cpp

   // On buffers
   buf.atomic(idx).fetch_add(1.0f);
   buf.atomic(idx).fetch_sub(1.0f);
   buf.atomic(idx).fetch_max(val);
   buf.atomic(idx).fetch_min(val);
   buf.atomic(idx).compare_exchange(expected, desired);

   // On vector components
   buf.atomic(idx).x.fetch_add(1.0f);

   // On struct members
   buf.atomic(idx).member_name.fetch_add(1.0f);

Kernel and Callable
^^^^^^^^^^^^^^^^^^^

``Kernel1D``, ``Kernel2D``, ``Kernel3D`` are entry points for GPU dispatch.
``Callable`` is a reusable device function.

.. code-block:: cpp

   // Callable — reusable function
   Callable square = [](Float x) noexcept { return x * x; };

   // Kernel — GPU entry point
   Kernel1D compute = [&](BufferFloat buf) noexcept {
       auto idx = dispatch_id().x;
       buf.write(idx, square(buf.read(idx)));
   };

   // Compile and dispatch
   auto shader = device.compile(compute);
   stream << shader(buffer).dispatch(1024);

**Callable captures:** Callables can capture buffers/images by reference.
Captures propagate transitively — a kernel that calls a callable inherits its captures.

**Multiple return values:**

.. code-block:: cpp

   Callable split = [](Float4 v) noexcept {
       return compose(v.xyz(), v.w);
   };

   // Unpack
   auto result = split(color);
   Float3 rgb = result.get<0>();
   Float alpha = result.get<1>();


DSL Built-in Functions
----------------------

Thread Queries
^^^^^^^^^^^^^^

.. code-block:: cpp

   UInt3 dispatch_id();      // Global thread position in the dispatch grid
   UInt3 dispatch_size();    // Total grid size
   UInt3 thread_id();        // Thread position within its block
   UInt3 block_id();         // Block position in the grid
   UInt3 block_size();       // Block dimensions

   // Convenience
   UInt dispatch_x();        // dispatch_id().x
   UInt thread_x();          // thread_id().x

Control Flow
^^^^^^^^^^^^

.. code-block:: cpp

   $if (cond) { ... };
   $if (cond) { ... } $elif (cond2) { ... } $else { ... };

   $while (cond) { ... };
   $for (i, count) { ... };           // 0 to count-1
   $for (i, begin, end) { ... };      // begin to end-1
   $for (i, begin, end, step) { ... };

   $loop { ... };                     // Infinite, use $break to exit
   $switch (val) { $case (v) { ... }; $default { ... }; };

   $break;
   $continue;
   $return(value);                    // Early return from callable

Type Conversions
^^^^^^^^^^^^^^^^

.. code-block:: cpp

   Float f = cast<float>(int_val);   // Static type cast (value conversion)
   UInt  u = as<uint>(float_val);    // Bitwise reinterpretation

Synchronization
^^^^^^^^^^^^^^^

.. code-block:: cpp

   sync_block();                     // Barrier for all threads in the block


Automatic Differentiation
-------------------------

LuisaCompute supports reverse-mode automatic differentiation via source-to-source
transformation inside ``$autodiff`` blocks:

.. code-block:: cpp

   $autodiff {
       requires_grad(x, y);          // Mark inputs as differentiable
       Float z = complex_function(x, y);
       backward(z);                  // Trigger backward pass
       Float dx = grad(x);           // Retrieve gradient
       Float dy = grad(y);
   };

**Supported features:**

- Control flow (``$if``/``$else``, ``$switch``)
- Callables (differentiation propagates through calls)
- Most built-in math functions

**Limitations:**

- Loops with dynamic iteration counts must be manually unrolled
- Some operations may not be differentiable


Custom Structures
-----------------

C++ structs must be registered with the ``LUISA_STRUCT`` macro to be usable in the DSL:

.. code-block:: cpp

   struct Material {
       float3 albedo;
       float roughness;
       float metallic;
   };

   // Register in global namespace — list all member fields
   LUISA_STRUCT(Material, albedo, roughness, metallic) {
       // Optional: DSL-side member functions
       [[nodiscard]] Float3 scaled_albedo(Float s) const noexcept {
           return albedo * s;
       }
   };

   // Use in kernels
   Kernel1D kernel = [&](BufferVar<Material> materials) noexcept {
       Var<Material> mat = materials.read(dispatch_x());
       Float3 color = mat->scaled_albedo(2.0f);
       mat.roughness = clamp(mat.roughness, 0.0f, 1.0f);
   };

**Template structs** use ``LUISA_TEMPLATE_STRUCT``:

.. code-block:: cpp

   template<typename K, typename V>
   struct Pair { K key; V value; };

   #define PAIR_TMPL() template<typename K, typename V>
   #define PAIR_TYPE() Pair<K, V>

   LUISA_TEMPLATE_STRUCT(PAIR_TMPL, PAIR_TYPE, key, value) {};

.. warning::

   - ``LUISA_STRUCT`` must be used in the **global namespace**.
   - Only scalar, vector, matrix, array, and already-registered struct members are allowed.
   - Whole-struct ``alignas`` up to 16 bytes is reflected; per-member ``alignas`` is not supported.


Sugar Syntax
------------

Include ``<luisa/dsl/sugar.h>`` for concise DSL macros:

.. code-block:: cpp

   #include <luisa/dsl/sugar.h>

   // Type shorthand
   $float x = 1.0f;             // Same as Float x = 1.0f
   $int i = 0;                  // Same as Int i = 0
   $float3 color;               // Same as Float3 color
   $ v = 10;                    // Auto-deduced: $int

   // Resource parameters with $ capture
   Kernel1D k = [&]($buffer<float> buf, $uint count) noexcept { ... };

   // Shared memory
   $shared<float> s{256};

   // Constants
   $constant data = {1.0f, 2.0f, 3.0f};


Logging
-------

LuisaCompute provides formatted logging via ``<luisa/core/logging.h>``:

.. code-block:: cpp

   #include <luisa/core/logging.h>

   LUISA_INFO("Processing {} items", count);
   LUISA_WARNING("Value {} exceeds limit", val);
   LUISA_VERBOSE("Debug: pos = ({}, {})", x, y);

   // Set log level
   luisa::log_level_info();         // Info and above
   luisa::log_level_verbose();      // All messages
   luisa::log_level_warning();      // Warnings and errors only

   // With source location
   LUISA_INFO_WITH_LOCATION("Checkpoint reached");

Uses `{fmt} <https://fmt.dev/>`_-style format strings.


Utility Classes
---------------

Clock
^^^^^

High-resolution timer (``<luisa/core/clock.h>``):

.. code-block:: cpp

   luisa::Clock clock;
   clock.tic();
   // ... work ...
   double ms = clock.toc();   // Elapsed milliseconds (does NOT reset)

BinaryBlob
^^^^^^^^^^

RAII wrapper for binary data (``<luisa/core/binary_io.h>``):

.. code-block:: cpp

   luisa::BinaryBlob blob{ptr, size, [](void *p) { ::operator delete(p); }};
   auto data = blob.data();
   auto sz   = blob.size();

DynamicModule
^^^^^^^^^^^^^

Cross-platform dynamic library loading (``<luisa/core/dynamic_module.h>``):

.. code-block:: cpp

   auto mod = luisa::DynamicModule::load("my_library");
   auto func = mod.function<int(int)>("my_function");
   int result = func(42);

Pool<T>
^^^^^^^

Fast object pool allocator with optional thread safety (``<luisa/core/pool.h>``):

.. code-block:: cpp

   luisa::Pool<MyClass> pool;           // Thread-safe
   luisa::Pool<MyClass, false> pool_nt; // Non-thread-safe

   MyClass *obj = pool.create(args...);
   pool.destroy(obj);
