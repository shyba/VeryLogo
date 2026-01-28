/**
 * Vulkan compute shader benchmark for tower field S-box with verification.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <fstream>
#include "aes_sbox_table.h"

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

int main(int argc, char** argv) {
    const char* spirv_path = argc > 1 ? argv[1] : "out/tower_sbox_spirv/circuit.spv";
    int iterations = argc > 2 ? atoi(argv[2]) : 10000;
    int num_threads = argc > 3 ? atoi(argv[3]) : 65536;

    printf("Tower Field S-box - Vulkan Benchmark with Verification\n");
    printf("=======================================================\n\n");
    printf("SPIR-V: %s\n", spirv_path);
    printf("Iterations: %d\n", iterations);
    printf("Threads: %d\n\n", num_threads);

    // Create instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VeryLogo";
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    CHECK_VK(vkCreateInstance(&createInfo, nullptr, &instance));

    // Find GPU
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    VkPhysicalDevice physicalDevice = devices[0];

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    printf("Device: %s\n\n", deviceProperties.deviceName);

    // Find compute queue
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    uint32_t computeQueueFamily = 0;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamily = i;
            break;
        }
    }

    // Create device
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

    // Load SPIR-V
    auto spirvCode = readFile(spirv_path);
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = spirvCode.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(spirvCode.data());

    VkShaderModule shaderModule;
    CHECK_VK(vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule));

    // Create descriptor set layout
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

    // Create pipeline layout and pipeline
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

    // Create buffers
    size_t bufferSize = num_threads * 8 * sizeof(uint32_t);
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VkBuffer inputBuffer, outputBuffer;
    CHECK_VK(vkCreateBuffer(device, &bufferInfo, nullptr, &inputBuffer));
    CHECK_VK(vkCreateBuffer(device, &bufferInfo, nullptr, &outputBuffer));

    // Allocate memory
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, inputBuffer, &memReqs);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    uint32_t memTypeIndex = 0;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    VkDeviceMemory inputMemory, outputMemory;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, nullptr, &inputMemory));
    CHECK_VK(vkAllocateMemory(device, &allocInfo, nullptr, &outputMemory));
    CHECK_VK(vkBindBufferMemory(device, inputBuffer, inputMemory, 0));
    CHECK_VK(vkBindBufferMemory(device, outputBuffer, outputMemory, 0));

    // Create descriptor pool and sets
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    VkDescriptorPool descriptorPool;
    CHECK_VK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));

    VkDescriptorSetAllocateInfo allocInfo2 = {};
    allocInfo2.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo2.descriptorPool = descriptorPool;
    allocInfo2.descriptorSetCount = 1;
    allocInfo2.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    CHECK_VK(vkAllocateDescriptorSets(device, &allocInfo2, &descriptorSet));

    VkDescriptorBufferInfo bufferInfos[2] = {};
    bufferInfos[0].buffer = inputBuffer;
    bufferInfos[0].range = bufferSize;
    bufferInfos[1].buffer = outputBuffer;
    bufferInfos[1].range = bufferSize;

    VkWriteDescriptorSet descriptorWrites[2] = {};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfos[0];

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &bufferInfos[1];

    vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, nullptr);

    // Create command pool and buffer
    VkCommandPoolCreateInfo poolInfo2 = {};
    poolInfo2.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo2.queueFamilyIndex = computeQueueFamily;

    VkCommandPool commandPool;
    CHECK_VK(vkCreateCommandPool(device, &poolInfo2, nullptr, &commandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    CHECK_VK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer));

    // Record command buffer
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    CHECK_VK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, (num_threads + 63) / 64, 1, 1);
    CHECK_VK(vkEndCommandBuffer(commandBuffer));

    // ============================================
    // VERIFICATION: Test all 256 S-box values
    // ============================================
    printf("Verifying correctness (all 256 S-box values)...\n");

    void* inputData;
    CHECK_VK(vkMapMemory(device, inputMemory, 0, bufferSize, 0, &inputData));
    uint32_t* inputWords = (uint32_t*)inputData;
    memset(inputWords, 0, bufferSize);

    // Set up bit-sliced input for values 0-255
    for (int thread = 0; thread < 8; thread++) {
        for (int bit = 0; bit < 8; bit++) {
            uint32_t bitplane = 0;
            for (int lane = 0; lane < 32; lane++) {
                int value = thread * 32 + lane;
                if ((value & (1 << bit))) {
                    bitplane |= (1u << lane);
                }
            }
            inputWords[thread * 8 + bit] = bitplane;
        }
    }
    vkUnmapMemory(device, inputMemory);

    // Run shader
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    CHECK_VK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    CHECK_VK(vkQueueWaitIdle(computeQueue));

    // Read back and verify
    void* outputData;
    CHECK_VK(vkMapMemory(device, outputMemory, 0, bufferSize, 0, &outputData));
    uint32_t* outputWords = (uint32_t*)outputData;

    int errors = 0;
    for (int thread = 0; thread < 8; thread++) {
        for (int lane = 0; lane < 32; lane++) {
            int input_val = thread * 32 + lane;

            // Extract output byte from bit-planes
            uint8_t output_byte = 0;
            for (int bit = 0; bit < 8; bit++) {
                uint32_t bitplane = outputWords[thread * 8 + bit];
                if (bitplane & (1u << lane)) {
                    output_byte |= (1 << bit);
                }
            }

            uint8_t expected = AES_SBOX[input_val];
            if (output_byte != expected) {
                if (errors < 10) {
                    printf("  ERROR at 0x%02x: got 0x%02x, expected 0x%02x\n",
                           input_val, output_byte, expected);
                }
                errors++;
            }
        }
    }
    vkUnmapMemory(device, outputMemory);

    if (errors > 0) {
        printf("  ❌ FAILED: %d errors out of 256\n", errors);
        return 1;
    }

    printf("  ✅ All 256 S-box values correct!\n\n");

    // ============================================
    // BENCHMARK
    // ============================================
    printf("Running benchmark...\n");

    // Reset input to benchmark pattern
    CHECK_VK(vkMapMemory(device, inputMemory, 0, bufferSize, 0, &inputData));
    memset(inputData, 0xAA, bufferSize);
    vkUnmapMemory(device, inputMemory);

    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        CHECK_VK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    }
    CHECK_VK(vkQueueWaitIdle(computeQueue));
    uint64_t end = get_time_ns();

    double elapsed_s = (end - start) / 1e9;
    uint64_t total_evals = (uint64_t)num_threads * 32 * iterations;
    double ns_per_eval = (end - start) / (double)total_evals;
    double evals_per_sec = total_evals / elapsed_s;

    printf("\nResults:\n");
    printf("========\n");
    printf("Device: %s\n", deviceProperties.deviceName);
    printf("Threads: %d\n", num_threads);
    printf("Iterations: %d\n", iterations);
    printf("Total evaluations: %lu (threads × 32 × iterations)\n", total_evals);
    printf("Total time: %.3f ms\n", elapsed_s * 1000);
    printf("Time per eval: %.3f ns\n", ns_per_eval);
    printf("Throughput: %.2f B evals/sec\n\n", evals_per_sec / 1e9);

    printf("✅ Verification: All 256 S-box values correct\n");
    printf("✅ Performance: %.2f billion evals/sec on AMD Radeon\n", evals_per_sec / 1e9);

    // Cleanup
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyBuffer(device, inputBuffer, nullptr);
    vkDestroyBuffer(device, outputBuffer, nullptr);
    vkFreeMemory(device, inputMemory, nullptr);
    vkFreeMemory(device, outputMemory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}
