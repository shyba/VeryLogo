/**
 * Generic Vulkan compute shader benchmark for VeryLogo circuits.
 *
 * Runs a SPIR-V compute shader with configurable input/output word counts.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <vector>
#include <fstream>
#include <string>

#define CHECK_VK(call) { \
    VkResult err = call; \
    if (err != VK_SUCCESS) { \
        fprintf(stderr, "Vulkan error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
}

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static std::vector<char> readFile(const char* filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

static const uint8_t AES_SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static inline uint8_t aes_xtime(uint8_t a) {
    uint8_t a1 = (uint8_t)((a << 1) & 0xFFu);
    return (uint8_t)(a1 ^ ((a & 0x80u) ? 0x1Bu : 0x00u));
}

static void aes_round_ref(const uint8_t st[16], const uint8_t rk[16], uint8_t out[16]) {
    uint8_t sb[16];
    uint8_t sr[16];
    uint8_t mc[16];
    for (int i = 0; i < 16; i++) {
        sb[i] = AES_SBOX[st[i]];
    }
    sr[0] = sb[0];   sr[1] = sb[5];   sr[2] = sb[10];  sr[3] = sb[15];
    sr[4] = sb[4];   sr[5] = sb[9];   sr[6] = sb[14];  sr[7] = sb[3];
    sr[8] = sb[8];   sr[9] = sb[13];  sr[10] = sb[2];  sr[11] = sb[7];
    sr[12] = sb[12]; sr[13] = sb[1];  sr[14] = sb[6];  sr[15] = sb[11];

    for (int c = 0; c < 4; c++) {
        int base = c * 4;
        uint8_t a0 = sr[base + 0];
        uint8_t a1 = sr[base + 1];
        uint8_t a2 = sr[base + 2];
        uint8_t a3 = sr[base + 3];
        uint8_t t = (uint8_t)(a0 ^ a1 ^ a2 ^ a3);
        uint8_t u = a0;
        mc[base + 0] = (uint8_t)(a0 ^ t ^ aes_xtime((uint8_t)(a0 ^ a1)));
        mc[base + 1] = (uint8_t)(a1 ^ t ^ aes_xtime((uint8_t)(a1 ^ a2)));
        mc[base + 2] = (uint8_t)(a2 ^ t ^ aes_xtime((uint8_t)(a2 ^ a3)));
        mc[base + 3] = (uint8_t)(a3 ^ t ^ aes_xtime((uint8_t)(a3 ^ u)));
    }
    for (int i = 0; i < 16; i++) {
        out[i] = (uint8_t)(mc[i] ^ rk[i]);
    }
}

static uint32_t find_memory_type(
    VkPhysicalDevice physicalDevice,
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties
) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

static void create_buffer(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer* buffer,
    VkDeviceMemory* bufferMemory
) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    CHECK_VK(vkCreateBuffer(device, &bufferInfo, nullptr, buffer));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, *buffer, &memReqs);

    uint32_t memTypeIndex = find_memory_type(
        physicalDevice,
        memReqs.memoryTypeBits,
        properties
    );
    if (memTypeIndex == UINT32_MAX) {
        fprintf(stderr, "No suitable memory type found\n");
        exit(1);
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    CHECK_VK(vkAllocateMemory(device, &allocInfo, nullptr, bufferMemory));
    CHECK_VK(vkBindBufferMemory(device, *buffer, *bufferMemory, 0));
}

int main(int argc, char** argv) {
    const char* spirv_path = argc > 1 ? argv[1] : "out/circuit.spv";
    int iterations = argc > 2 ? atoi(argv[2]) : 1000;
    int num_threads = argc > 3 ? atoi(argv[3]) : 65536;
    int local_x = argc > 4 ? atoi(argv[4]) : 64;
    int local_y = argc > 5 ? atoi(argv[5]) : 1;
    int local_z = argc > 6 ? atoi(argv[6]) : 1;
    int input_words = argc > 7 ? atoi(argv[7]) : 8;
    int output_words = argc > 8 ? atoi(argv[8]) : 8;
    int rounds = argc > 9 ? atoi(argv[9]) : 1;
    int lanes = argc > 10 ? atoi(argv[10]) : 32;
    int bytes_per_eval = argc > 11 ? atoi(argv[11]) : 0;
    int check = argc > 12 ? atoi(argv[12]) : 0;
    int rk_off = argc > 13 ? atoi(argv[13]) : 0;
    int out_off = argc > 14 ? atoi(argv[14]) : 0;
    int width_bits = argc > 15 ? atoi(argv[15]) : 0;
    int check_threads = argc > 16 ? atoi(argv[16]) : 1;
    int dump_words = argc > 17 ? atoi(argv[17]) : 0;
    int dump_offset = argc > 18 ? atoi(argv[18]) : 0;
    int kernel_steps = argc > 19 ? atoi(argv[19]) : 1;
    int toggle_sets = argc > 20 ? atoi(argv[20]) : 1;
    int st_off = argc > 21 ? atoi(argv[21]) : -1;
    int state_out_off = argc > 22 ? atoi(argv[22]) : -1;
    if (kernel_steps < 1) kernel_steps = 1;
    if (input_words <= 0 || output_words <= 0 || num_threads <= 0) {
        fprintf(stderr, "invalid dimensions: input_words/output_words/threads must be > 0\n");
        return 1;
    }

    int local_size = local_x * local_y * local_z;
    if (local_size <= 0) {
        fprintf(stderr, "invalid local size\n");
        return 1;
    }
    int groups = (num_threads + local_size - 1) / local_size;
    int effective_threads = groups * local_size;
    if (effective_threads <= 0) {
        fprintf(stderr, "invalid effective thread count\n");
        return 1;
    }

    printf("VeryLogo Vulkan Benchmark\n");
    printf("=========================\n\n");
    printf("SPIR-V file: %s\n", spirv_path);
    printf("Iterations: %d\n", iterations);
    printf("Threads: %d (effective %d)\n", num_threads, effective_threads);
    printf("LocalSize: %d %d %d\n", local_x, local_y, local_z);
    printf("Input words: %d\n", input_words);
    printf("Output words: %d\n", output_words);
    printf("Rounds: %d\n", rounds);
    printf("Kernel steps: %d\n", kernel_steps);
    printf("Toggle descriptor sets: %s\n", toggle_sets ? "yes" : "no");
    printf("Lanes per thread: %d\n", lanes);
    printf("\n");

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VeryLogo Benchmark";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "VeryLogo";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    CHECK_VK(vkCreateInstance(&createInfo, nullptr, &instance));

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        fprintf(stderr, "No Vulkan devices found\n");
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    auto score_device = [](const VkPhysicalDeviceProperties& props) -> int {
        int score = 0;
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                score = 300;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                score = 200;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                score = 100;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                score = 0;
                break;
            default:
                score = 50;
                break;
        }
        std::string name(props.deviceName);
        if (name.find("llvmpipe") != std::string::npos) score -= 1000;
        return score;
    };

    VkPhysicalDevice physicalDevice = devices[0];
    VkPhysicalDeviceProperties bestProps = {};
    vkGetPhysicalDeviceProperties(physicalDevice, &bestProps);
    int bestScore = score_device(bestProps);
    for (uint32_t i = 1; i < deviceCount; i++) {
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(devices[i], &props);
        int sc = score_device(props);
        if (sc > bestScore) {
            bestScore = sc;
            bestProps = props;
            physicalDevice = devices[i];
        }
    }

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    printf("Using device: %s\n\n", deviceProperties.deviceName);

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, &queueFamilyCount, queueFamilies.data()
    );

    uint32_t computeQueueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamily = i;
            break;
        }
    }
    if (computeQueueFamily == UINT32_MAX) {
        fprintf(stderr, "No compute queue family found\n");
        return 1;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    VkDevice device;
    CHECK_VK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    VkQueue computeQueue;
    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

    auto spirvCode = readFile(spirv_path);
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = spirvCode.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(spirvCode.data());

    VkShaderModule shaderModule;
    CHECK_VK(vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule));

    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo = {};
    descriptorLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutInfo.bindingCount = 2;
    descriptorLayoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout pipelineLayout;
    CHECK_VK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    CHECK_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

    VkDeviceSize inputSize = (VkDeviceSize)effective_threads * input_words * sizeof(uint32_t);
    VkDeviceSize outputSize = (VkDeviceSize)effective_threads * output_words * sizeof(uint32_t);

    if (rounds > 1 && output_words < input_words) {
        fprintf(stderr, "rounds>1 requires output_words >= input_words for ping-pong\n");
        return 1;
    }
    if (toggle_sets && output_words < input_words) {
        fprintf(stderr, "toggle_sets=1 requires output_words >= input_words\n");
        return 1;
    }

    VkBuffer bufA, bufB;
    VkDeviceMemory memA, memB;
    create_buffer(
        device, physicalDevice, inputSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &bufA, &memA
    );
    create_buffer(
        device, physicalDevice, outputSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &bufB, &memB
    );

    auto pattern_word = [](uint32_t i) -> uint32_t {
        return 0xA5A50000u + i;
    };

    std::vector<uint32_t> input_shadow((size_t)effective_threads * input_words);
    for (size_t i = 0; i < input_shadow.size(); i++) {
        input_shadow[i] = pattern_word((uint32_t)i);
    }

    auto init_buffer = [&](VkDeviceMemory mem, VkDeviceSize size, uint32_t fill, bool seed_inputs) {
        void* data = nullptr;
        CHECK_VK(vkMapMemory(device, mem, 0, size, 0, &data));
        uint32_t* words = reinterpret_cast<uint32_t*>(data);
        size_t total_words = (size_t)(size / sizeof(uint32_t));
        for (size_t i = 0; i < total_words; i++) {
            words[i] = fill;
        }
        if (seed_inputs) {
            size_t per_thread_words = total_words / (size_t)effective_threads;
            if ((size_t)input_words > per_thread_words) {
                fprintf(stderr,
                        "seed_inputs overflow: input_words=%d exceeds mapped words/thread=%zu\n",
                        input_words, per_thread_words);
                vkUnmapMemory(device, mem);
                exit(1);
            }
            for (int t = 0; t < effective_threads; t++) {
                size_t base = (size_t)t * input_words;
                for (int b = 0; b < input_words; b++) {
                    words[base + b] = input_shadow[base + b];
                }
            }
        }
        vkUnmapMemory(device, mem);
    };

    auto init_pingpong_inputs = [&](bool strict_check_mode) {
        if (strict_check_mode) {
            // For correctness checks, poison output-side memory so partial writes
            // cannot accidentally look correct.
            init_buffer(memA, inputSize, 0xDEADBEEFu, false);
            init_buffer(memB, outputSize, 0xDEADBEEFu, false);
            // Descriptor set 0 reads from A first, so A must contain valid inputs.
            init_buffer(memA, inputSize, 0u, true);
        } else {
            init_buffer(memA, inputSize, 0u, true);
            init_buffer(memB, outputSize, 0u, true);
        }
    };

    bool strict_check_mode = (check != 0);
    init_pingpong_inputs(strict_check_mode);

    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 4;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 2;

    VkDescriptorPool descriptorPool;
    CHECK_VK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));

    VkDescriptorSetLayout layouts[2] = { descriptorSetLayout, descriptorSetLayout };
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts = layouts;

    VkDescriptorSet descriptorSets[2];
    CHECK_VK(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets));

    VkDescriptorBufferInfo inputInfoA = { bufA, 0, inputSize };
    VkDescriptorBufferInfo outputInfoB = { bufB, 0, outputSize };
    VkDescriptorBufferInfo inputInfoB = { bufB, 0, outputSize };
    VkDescriptorBufferInfo outputInfoA = { bufA, 0, inputSize };

    VkWriteDescriptorSet writes[4] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptorSets[0];
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &inputInfoA;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptorSets[0];
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &outputInfoB;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptorSets[1];
    writes[2].dstBinding = 0;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &inputInfoB;
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = descriptorSets[1];
    writes[3].dstBinding = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].descriptorCount = 1;
    writes[3].pBufferInfo = &outputInfoA;

    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

    VkCommandPoolCreateInfo commandPoolInfo = {};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex = computeQueueFamily;

    VkCommandPool commandPool;
    CHECK_VK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 2;

    VkCommandBuffer commandBuffers[2];
    CHECK_VK(vkAllocateCommandBuffers(device, &cmdAllocInfo, commandBuffers));

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    CHECK_VK(vkCreateFence(device, &fenceInfo, nullptr, &fence));

    auto record_dispatch = [&](VkCommandBuffer commandBuffer, int start_set) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        CHECK_VK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        int set_idx = start_set;
        for (int r = 0; r < rounds; r++) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelineLayout,
                0,
                1,
                &descriptorSets[set_idx],
                0,
                nullptr
            );
            vkCmdDispatch(commandBuffer, groups, 1, 1);

            VkBufferMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.buffer = (set_idx == 0) ? bufB : bufA;
            barrier.offset = 0;
            barrier.size = (set_idx == 0) ? outputSize : inputSize;

            vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &barrier,
                0, nullptr
            );
            set_idx ^= 1;
        }
        CHECK_VK(vkEndCommandBuffer(commandBuffer));
        return set_idx;
    };

    auto submit_and_wait = [&](VkCommandBuffer commandBuffer) {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        CHECK_VK(vkQueueSubmit(computeQueue, 1, &submitInfo, fence));
        CHECK_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        CHECK_VK(vkResetFences(device, 1, &fence));
    };

    int end_set0 = record_dispatch(commandBuffers[0], 0);
    int end_set1 = record_dispatch(commandBuffers[1], 1);
    if (end_set0 != (0 ^ (rounds & 1)) || end_set1 != (1 ^ (rounds & 1))) {
        fprintf(stderr, "Unexpected descriptor-set parity after recording\n");
        return 1;
    }

    int set_toggle = toggle_sets ? (rounds & 1) : 0;
    int start_set = 0;
    int warmup_iters = check ? 0 : 5;
    for (int i = 0; i < warmup_iters; i++) {
        submit_and_wait(commandBuffers[start_set]);
        start_set ^= set_toggle;
    }

    // Ensure timed and checked runs always start from known input state.
    init_pingpong_inputs(strict_check_mode);
    start_set = 0;

    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        submit_and_wait(commandBuffers[start_set]);
        start_set ^= set_toggle;
    }
    uint64_t end = get_time_ns();

    double elapsed = (double)(end - start) / 1e9;
    double evals =
        (double)iterations *
        (double)rounds *
        (double)kernel_steps *
        (double)effective_threads *
        (double)lanes;
    double ns_per_eval = elapsed * 1e9 / evals;
    double evals_per_sec = evals / elapsed;

    printf("Elapsed: %.6f s\n", elapsed);
    printf("Time per eval: %.3f ns\n", ns_per_eval);
    printf("Throughput: %.3fB evals/sec\n", evals_per_sec / 1e9);
    if (bytes_per_eval > 0) {
        double mib_s = evals_per_sec * (double)bytes_per_eval / (1024.0 * 1024.0);
        printf("Throughput: %.2f MiB/s\n", mib_s);
    }
    if (check) {
        if (width_bits < 0 || rk_off < 0 || out_off < 0) {
            fprintf(stderr, "invalid check params: width_bits/rk_off/out_off must be non-negative\n");
            return 1;
        }
        if ((size_t)(rk_off + width_bits) > (size_t)input_words) {
            fprintf(stderr,
                    "invalid check params: rk_off(%d)+width_bits(%d) exceeds input_words(%d)\n",
                    rk_off, width_bits, input_words);
            return 1;
        }
        if ((size_t)(out_off + width_bits) > (size_t)output_words) {
            fprintf(stderr,
                    "invalid check params: out_off(%d)+width_bits(%d) exceeds output_words(%d)\n",
                    out_off, width_bits, output_words);
            return 1;
        }
        int threads_to_check = check_threads;
        if (threads_to_check < 1) threads_to_check = effective_threads;
        if (threads_to_check > effective_threads) threads_to_check = effective_threads;

        auto count_mismatches = [&](VkDeviceMemory mem, const char* label) {
            void* out_data = nullptr;
            CHECK_VK(vkMapMemory(device, mem, 0, VK_WHOLE_SIZE, 0, &out_data));
            uint32_t* outWords = reinterpret_cast<uint32_t*>(out_data);
            int mismatches = 0;
            for (int t = 0; t < threads_to_check; t++) {
                size_t in_base = (size_t)t * input_words;
                size_t out_base = (size_t)t * output_words;
                for (int b = 0; b < width_bits; b++) {
                    uint32_t expected = input_shadow[in_base + rk_off + b];
                    uint32_t got = outWords[out_base + out_off + b];
                    if (expected != got) {
                        mismatches++;
                        if (mismatches < 3) {
                            fprintf(stderr, "[%s] t=%d b=%d exp=0x%08x got=0x%08x\n",
                                    label, t, b, expected, got);
                        }
                    }
                }
            }
            if (dump_words > 0) {
                int dump_count = dump_words;
                if (dump_count > output_words) dump_count = output_words;
                int start = dump_offset;
                if (start < 0) start = 0;
                if (start >= output_words) start = output_words - 1;
                if (start + dump_count > output_words) dump_count = output_words - start;
                fprintf(stdout, "DUMP_%s thread0 words [%d..%d):\n", label, start, start + dump_count);
                size_t out_base = 0;
                for (int i = 0; i < dump_count; i++) {
                    int idx = start + i;
                    fprintf(stdout, "  [%d]=0x%08x\n", idx, outWords[out_base + idx]);
                }
            }
            vkUnmapMemory(device, mem);
            return mismatches;
        };

        int last_start_set = 0;
        if (iterations > 0) {
            // After loop, start_set points to the next command-buffer index.
            last_start_set = start_set ^ set_toggle;
        }
        int end_set_last = (last_start_set == 0) ? end_set0 : end_set1;
        int final_set_idx = end_set_last ^ 1;
        bool final_is_B = (final_set_idx == 0);

        VkDeviceMemory final_mem = final_is_B ? memB : memA;
        VkDeviceMemory nonfinal_mem = final_is_B ? memA : memB;
        const char* final_label = final_is_B ? "B" : "A";
        const char* nonfinal_label = final_is_B ? "A" : "B";

        int mism_final = count_mismatches(final_mem, final_label);
        int mism_nonfinal = count_mismatches(nonfinal_mem, nonfinal_label);
        printf("Check mismatches: final(%s)=%d nonfinal(%s)=%d\n",
               final_label, mism_final, nonfinal_label, mism_nonfinal);
        if (mism_final == 0) {
            printf("Check: rk pass-through OK on final buffer (%d threads, %d bits)\n",
                   threads_to_check, width_bits);
        } else {
            fprintf(stderr, "Check failed on final buffer %s: %d mismatches\n",
                    final_label, mism_final);
            return 2;
        }

        bool can_check_aes = (
            st_off >= 0 &&
            state_out_off >= 0 &&
            width_bits == 128 &&
            rounds == 1 &&
            kernel_steps == 1
        );
        if (can_check_aes) {
            if ((size_t)(st_off + 128) > (size_t)input_words ||
                (size_t)(state_out_off + 128) > (size_t)output_words) {
                fprintf(stderr,
                        "invalid AES check params: st/state_out offsets exceed buffer words\n");
                return 1;
            }
        }
        if (can_check_aes) {
            int lanes_to_check = lanes;
            if (lanes_to_check < 1) lanes_to_check = 1;
            if (lanes_to_check > 32) lanes_to_check = 32;

            void* out_data = nullptr;
            CHECK_VK(vkMapMemory(device, final_mem, 0, VK_WHOLE_SIZE, 0, &out_data));
            uint32_t* outWords = reinterpret_cast<uint32_t*>(out_data);

            int aes_mismatches = 0;
            for (int t = 0; t < threads_to_check; t++) {
                size_t in_base = (size_t)t * input_words;
                size_t out_base = (size_t)t * output_words;
                for (int lane = 0; lane < lanes_to_check; lane++) {
                    uint8_t st_bytes[16] = {};
                    uint8_t rk_bytes[16] = {};
                    uint8_t ref_bytes[16] = {};
                    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
                        uint8_t stv = 0;
                        uint8_t rkv = 0;
                        int bit_base = (15 - byte_idx) * 8;
                        for (int bit = 0; bit < 8; bit++) {
                            int bit_idx = bit_base + bit;
                            uint32_t st_word = input_shadow[in_base + st_off + bit_idx];
                            uint32_t rk_word = input_shadow[in_base + rk_off + bit_idx];
                            stv |= (uint8_t)(((st_word >> lane) & 1u) << bit);
                            rkv |= (uint8_t)(((rk_word >> lane) & 1u) << bit);
                        }
                        st_bytes[byte_idx] = stv;
                        rk_bytes[byte_idx] = rkv;
                    }
                    aes_round_ref(st_bytes, rk_bytes, ref_bytes);
                    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
                        int bit_base = (15 - byte_idx) * 8;
                        uint8_t ref = ref_bytes[byte_idx];
                        for (int bit = 0; bit < 8; bit++) {
                            int bit_idx = bit_base + bit;
                            uint32_t got_word = outWords[out_base + state_out_off + bit_idx];
                            uint32_t got = (got_word >> lane) & 1u;
                            uint32_t exp = (uint32_t)((ref >> bit) & 1u);
                            if (got != exp) {
                                aes_mismatches++;
                                if (aes_mismatches <= 3) {
                                    fprintf(
                                        stderr,
                                        "[AES] t=%d lane=%d byte=%d bit=%d exp=%u got=%u\n",
                                        t, lane, byte_idx, bit, exp, got
                                    );
                                }
                            }
                        }
                    }
                }
            }
            vkUnmapMemory(device, final_mem);
            if (aes_mismatches == 0) {
                printf(
                    "Check: AES round output OK on final buffer (%d threads, %d lanes/thread)\n",
                    threads_to_check, lanes_to_check
                );
            } else {
                fprintf(stderr, "Check failed: AES round mismatches=%d\n", aes_mismatches);
                return 3;
            }
        }
    }
    return 0;
}
