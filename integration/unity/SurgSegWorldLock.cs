using UnityEngine;

/// World-locked target/danger overlay. A model result arrives tens of ms to ~1 s after its frame, but the view (head /
/// camera) moves every frame. When a frame is sent to the model we also store its depth and
/// camera pose; when the result comes back, every rendered frame re-projects each visible
/// surface point into that stored view and takes its role colour from there. The highlight
/// therefore stays on the tissue at full frame rate while the camera moves, with no lag.
/// Needs only a depth buffer and the camera pose, so it works in any Unity scene.
/// Added automatically by SurgSegClient to its camera.
[RequireComponent(typeof(Camera))]
public class SurgSegWorldLock : MonoBehaviour
{
    public static bool capturing;                 // true while the client renders frames for the model
    public Shader reprojectShader, eyeDepthShader;
    public bool active = true;
    [Range(0, 1)] public float opacity = 1f;
    public float depthTolerance = 0.004f;         // metres; also 3% of depth (see shader)

    Camera cam;
    Material mat;
    readonly RenderTexture[] depth = new RenderTexture[2];
    readonly Matrix4x4[] view = new Matrix4x4[2], proj = new Matrix4x4[2];
    int shown = -1;
    Texture roleTex;

    public bool HasOverlay => shown >= 0 && roleTex != null;

    void OnEnable()
    {
        cam = GetComponent<Camera>();
        cam.depthTextureMode |= DepthTextureMode.Depth;
        if (reprojectShader == null) reprojectShader = Shader.Find("Hidden/SurgSeg/Reproject");
        if (eyeDepthShader == null) eyeDepthShader = Shader.Find("Hidden/SurgSeg/EyeDepth");
    }

    /// Render the camera's eye depth for the frame being sent to the model. Returns the slot
    /// to pass to Show() when that frame's result arrives.
    public int CaptureDepth(int width, int height)
    {
        int slot = shown == 0 ? 1 : 0;            // never overwrite the slot on screen
        if (depth[slot] == null || depth[slot].width != width || depth[slot].height != height)
        {
            if (depth[slot] != null) depth[slot].Release();
            depth[slot] = new RenderTexture(width, height, 24, RenderTextureFormat.RFloat)
                { filterMode = FilterMode.Point, name = "SurgSegCaptureDepth" + slot };
        }
        var prevTarget = cam.targetTexture;
        var prevClear = cam.clearFlags;
        var prevBg = cam.backgroundColor;
        capturing = true;
        cam.targetTexture = depth[slot];
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.clear;        // 0 = no surface
        cam.RenderWithShader(eyeDepthShader, "");
        view[slot] = cam.worldToCameraMatrix;
        proj[slot] = cam.projectionMatrix;
        cam.targetTexture = prevTarget;
        cam.clearFlags = prevClear;
        cam.backgroundColor = prevBg;
        capturing = false;
        return slot;
    }

    public void Show(int slot, Texture role) { shown = slot; roleTex = role; }
    public void Hide() { shown = -1; }

    void OnRenderImage(RenderTexture src, RenderTexture dst)
    {
        if (capturing || !active || !HasOverlay || reprojectShader == null)
        {
            Graphics.Blit(src, dst);
            return;
        }
        if (mat == null) mat = new Material(reprojectShader) { hideFlags = HideFlags.HideAndDontSave };
        mat.SetTexture("_RoleTex", roleTex);
        mat.SetTexture("_CapDepth", depth[shown]);
        mat.SetMatrix("_CapView", view[shown]);
        mat.SetMatrix("_CapVP", proj[shown] * view[shown]);
        mat.SetMatrix("_InvProj", cam.projectionMatrix.inverse);
        mat.SetMatrix("_CamToWorld", cam.cameraToWorldMatrix);
        mat.SetFloat("_FarClip", cam.farClipPlane);
        mat.SetFloat("_DepthTol", depthTolerance);
        mat.SetFloat("_Opacity", opacity);
        Graphics.Blit(src, dst, mat);
    }

    void OnDestroy()
    {
        foreach (var d in depth) if (d != null) d.Release();
        if (mat != null) DestroyImmediate(mat);
    }
}
