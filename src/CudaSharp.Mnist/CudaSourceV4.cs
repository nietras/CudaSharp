namespace CudaSharp.Mnist;

public static partial class Program
{
    public static string CudaSourceV4 => CudaSourceV1
        .Replace("\r\n", "\n")
        .Replace("#define BATCH_SIZE 128", "#define BATCH_SIZE 128")
        .Replace("#define TOTAL_STEPS 400", "#define TOTAL_STEPS 190")
        .Replace("float max_lr = MAX_LR;", "float max_lr = MAX_LR;")
        .Replace("float beta1 = 0.7f;", "float beta1 = 0.9f;")
        .Replace("float beta2 = 0.9f;", "float beta2 = 0.999f;")
        .Replace(@"            float pct = (float)step_val / total_steps;
            float start_lr = max_lr / 25.0f;
            float end_lr = max_lr / 1000.0f;
            float peak_pct = 0.3f;
            
            float lr = 0.0f;
            if (pct < peak_pct)
            {
                float phase_pct = pct / peak_pct;
                float cos_val = cosf(3.14159265f * phase_pct);
                lr = start_lr + 0.5f * (max_lr - start_lr) * (1.0f - cos_val);
            }
            else
            {
                float phase_pct = (pct - peak_pct) / (1.0f - peak_pct);
                float cos_val = cosf(3.14159265f * phase_pct);
                lr = end_lr + 0.5f * (max_lr - end_lr) * (1.0f + cos_val);
            }", @"            float lr = 0.0f;
            int decay_start = (int)(total_steps * 0.75f);
            if (step_val < decay_start)
            {
                lr = max_lr;
            }
            else
            {
                float phase_pct = (float)(step_val - decay_start) / (total_steps - decay_start);
                float cos_val = cosf(3.14159265f * phase_pct);
                lr = max_lr * 0.5f * (1.0f + cos_val);
            }");
}
