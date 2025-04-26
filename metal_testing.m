#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

uint64_t now_nsec() {
    static mach_timebase_info_data_t tb = {0,0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t t = mach_absolute_time();
    // convert to nanoseconds
    return t * tb.numer / tb.denom;
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Creating device instance client
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "ERROR: Metal isn't being supported?\n");
            return -1;
        }

        // Loading & compiling matrix_multiple.metal file
        NSError *error = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"matrix_multiply.metal"
                                                  encoding:NSUTF8StringEncoding
                                                     error:&error];
        if (!src) {
            fprintf(stderr, "ERROR reading shader: %s\n", [[error localizedDescription] UTF8String]);
            return -1;
        }
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&error];
        if (!lib) {
            fprintf(stderr, "ERROR compiling shader: %s\n", [[error localizedDescription] UTF8String]);
            return -1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"matrixMultiply"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            fprintf(stderr, "ERROR creating pipeline: %s\n", [[error localizedDescription] UTF8String]);
            return -1;
        }

        // 3) Prepare input data
        const uint N = 1024;                              // matrix size N×N
        size_t count = (size_t)N * N;
        size_t bytes = count * sizeof(float);

        float *A = malloc(bytes), *B = malloc(bytes);
        for (size_t i = 0; i < count; ++i) {
            A[i] = (float)arc4random() / UINT32_MAX;
            B[i] = (float)arc4random() / UINT32_MAX;
        }

        id<MTLBuffer> bufA = [device newBufferWithBytes:A length:bytes options:0];
        id<MTLBuffer> bufB = [device newBufferWithBytes:B length:bytes options:0];
        id<MTLBuffer> bufC = [device newBufferWithLength:bytes options:0];
        id<MTLBuffer> bufN = [device newBufferWithBytes:&N length:sizeof(N) options:0];

        // 4) Create command queue & buffer
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBuffer:bufN offset:0 atIndex:3];

        // Dispatch step: one thread per matrix element
        MTLSize grid = MTLSizeMake(count, 1, 1);
        // threadExecutionWidth = optimal threadsPerThreadgroup
        NSUInteger w = pipeline.threadExecutionWidth;
        MTLSize tg = MTLSizeMake(w, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        // comitting and waiting till a completion signal
        uint64_t t0 = now_nsec();
        [cmd commit];
        [cmd waitUntilCompleted];
        uint64_t t1 = now_nsec();

        double elapsed_ms = (t1 - t0) / 1e6;
        printf("Matrix-multiply %ux%u on GPU took %.2f ms\n", N, N, elapsed_ms);

        // printing element [0]
        float *C = bufC.contents;
        printf("C[0] = %f\n", C[0]);

        free(A);
        free(B);
    }
    return 0;
}
