/**
 * Simple Vulkan test - just check S-box(0x00) = 0x63
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include "aes_sbox_table.h"

#define CHECK_VK(call) { \
    VkResult err = call; \
    if (err != VK_SUCCESS) { \
        fprintf(stderr, "Vulkan error: %d\n", err); \
        exit(1); \
    } \
}

static std::vector<char> readFile(const char* filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

int main(int argc, char** argv) {
    const char* spirv_path = argc > 1 ? argv[1] : "out/tower_sbox_spirv/compute_final.spv";

    printf("Simple Vulkan S-box test\n");
    printf("Testing: S-box(0x00) should equal 0x63\n\n");

    // Minimal Vulkan setup
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    VkInstance instance;
    CHECK_VK(vkCreateInstance(&ci, nullptr, &instance));

    std::vector<VkPhysicalDevice> devices(10);
    uint32_t count = 10;
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    VkPhysicalDevice pd = devices[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("GPU: %s\n\n", props.deviceName);

    // Find compute queue
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfProps.data());

    uint32_t qf = 0;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qf = i;
            break;
        }
    }

    float qp = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &qp;

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkDevice device;
    CHECK_VK(vkCreateDevice(pd, &dci, nullptr, &device));

    VkQueue queue;
    vkGetDeviceQueue(device, qf, 0, &queue);

    // Load shader
    auto code = readFile(spirv_path);
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size();
    smi.pCode = (const uint32_t*)code.data();

    VkShaderModule shader;
    CHECK_VK(vkCreateShaderModule(device, &smi, nullptr, &shader));

    // Descriptor set layout
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1] = bindings[0];
    bindings[1].binding = 1;

    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2;
    dslci.pBindings = bindings;

    VkDescriptorSetLayout dsl;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl));

    // Pipeline
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;

    VkPipelineLayout pl;
    CHECK_VK(vkCreatePipelineLayout(device, &plci, nullptr, &pl));

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = "main";
    cpci.layout = pl;

    VkPipeline pipeline;
    CHECK_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));

    // Buffers - just for one thread (8 inputs, 8 outputs)
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 64 * sizeof(uint32_t);  // 64 threads * 8 words
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VkBuffer inBuf, outBuf;
    CHECK_VK(vkCreateBuffer(device, &bci, nullptr, &inBuf));
    CHECK_VK(vkCreateBuffer(device, &bci, nullptr, &outBuf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, inBuf, &mr);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    uint32_t mti = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1 << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            mti = i;
            break;
        }
    }

    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mti;

    VkDeviceMemory inMem, outMem;
    CHECK_VK(vkAllocateMemory(device, &mai, nullptr, &inMem));
    CHECK_VK(vkAllocateMemory(device, &mai, nullptr, &outMem));
    CHECK_VK(vkBindBufferMemory(device, inBuf, inMem, 0));
    CHECK_VK(vkBindBufferMemory(device, outBuf, outMem, 0));

    // Initialize input: all zeros (S-box(0x00))
    void* data;
    CHECK_VK(vkMapMemory(device, inMem, 0, 64 * sizeof(uint32_t), 0, &data));
    uint32_t* words = (uint32_t*)data;
    memset(words, 0, 64 * sizeof(uint32_t));
    printf("Input bit-planes for S-box(0x00):\n");
    for (int i = 0; i < 8; i++) {
        printf("  Bit %d: 0x%08x\n", i, words[i]);
    }
    vkUnmapMemory(device, inMem);

    // Descriptor pool
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    dpci.maxSets = 1;

    VkDescriptorPool dp;
    CHECK_VK(vkCreateDescriptorPool(device, &dpci, nullptr, &dp));

    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;

    VkDescriptorSet ds;
    CHECK_VK(vkAllocateDescriptorSets(device, &dsai, &ds));

    VkDescriptorBufferInfo bis[2] = {};
    bis[0].buffer = inBuf;
    bis[0].range = VK_WHOLE_SIZE;
    bis[1].buffer = outBuf;
    bis[1].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet wds[2] = {};
    wds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds[0].dstSet = ds;
    wds[0].dstBinding = 0;
    wds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds[0].descriptorCount = 1;
    wds[0].pBufferInfo = &bis[0];

    wds[1] = wds[0];
    wds[1].dstBinding = 1;
    wds[1].pBufferInfo = &bis[1];

    vkUpdateDescriptorSets(device, 2, wds, 0, nullptr);

    // Command buffer
    VkCommandPoolCreateInfo cpci2 = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.queueFamilyIndex = qf;

    VkCommandPool cp;
    CHECK_VK(vkCreateCommandPool(device, &cpci2, nullptr, &cp));

    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;

    VkCommandBuffer cb;
    CHECK_VK(vkAllocateCommandBuffers(device, &cbai, &cb));

    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK_VK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    vkCmdDispatch(cb, 1, 1, 1);  // Just 1 workgroup = 64 threads
    CHECK_VK(vkEndCommandBuffer(cb));

    // Execute
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    CHECK_VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    CHECK_VK(vkQueueWaitIdle(queue));

    // Read results
    CHECK_VK(vkMapMemory(device, outMem, 0, 64 * sizeof(uint32_t), 0, &data));
    words = (uint32_t*)data;

    printf("\nOutput bit-planes:\n");
    for (int i = 0; i < 8; i++) {
        printf("  Bit %d: 0x%08x\n", i, words[i]);
    }

    // Extract S-box(0x00) from lane 0 of bit-planes
    uint8_t result = 0;
    for (int bit = 0; bit < 8; bit++) {
        if (words[bit] & 1) {  // Lane 0
            result |= (1 << bit);
        }
    }

    printf("\nS-box(0x00) = 0x%02x (expected 0x63)\n", result);
    if (result == 0x63) {
        printf("✅ Correct!\n");
    } else {
        printf("❌ Wrong! Binary: ");
        for (int i = 7; i >= 0; i--) {
            printf("%d", (result >> i) & 1);
        }
        printf("\nExpected binary: 01100011\n");
    }

    vkUnmapMemory(device, outMem);

    // Cleanup
    vkDestroyCommandPool(device, cp, nullptr);
    vkDestroyDescriptorPool(device, dp, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pl, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkDestroyBuffer(device, inBuf, nullptr);
    vkDestroyBuffer(device, outBuf, nullptr);
    vkFreeMemory(device, inMem, nullptr);
    vkFreeMemory(device, outMem, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return (result == 0x63) ? 0 : 1;
}
