// Records (camera frame, label mask) pairs from the endoscope camera, for fine-tuning
// surgseg on YOUR simulator's look. TEMPLATE: not compiled in this repo.
//
// Setup:
//   1. Put SurgSegLabel on tissue/tool objects (liver, gallbladder, artery, duct, tools...).
//   2. Add this component anywhere; assign the endoscope camera and SurgSegId.shader.
//   3. Play, press F9 to start/stop recording (one pair every `interval` seconds).
//      Move the camera/tools around and vary views - variety matters more than count.
// Output (exactly what surgseg.datasets.prepare_custom reads):
//   <outputRoot>/<episode>/images/000000.png    the frame the trainee sees
//   <outputRoot>/<episode>/masks/000000.png     pixel value = class id (SurgClass)
// Then copy <outputRoot> to the training machine and run prepare_custom with integration/unity/sim_labels.yaml
// (or zip it and run scripts/linux/finetune_sim.sh).
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class SurgSegCapture : MonoBehaviour
{
    public Camera endoscopeCamera;
    public Shader idShader;                       // assign SurgSegId.shader
    public string outputRoot = "SurgSegCapture";  // relative to the project folder
    public string episode = "ep01";               // use a new name per recording session
    public int width = 1280, height = 1024;
    public float interval = 0.5f;                 // seconds between saved pairs
    public KeyCode toggleKey = KeyCode.F9;

    bool recording;
    float next;
    int index;
    RenderTexture rgbRT, idRT;
    Texture2D rgbTex, idTex;
    readonly Dictionary<int, Material> idMats = new Dictionary<int, Material>();

    void Start()
    {
        if (idShader == null) idShader = Shader.Find("Hidden/SurgSegId");
        if (idShader == null) { Debug.LogError("SurgSegCapture: assign SurgSegId.shader"); enabled = false; return; }
        rgbRT = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        // Linear + no MSAA + point filtering: ids must come out bit-exact, never blended.
        idRT = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
               { antiAliasing = 1, filterMode = FilterMode.Point };
        rgbTex = new Texture2D(width, height, TextureFormat.RGB24, false);
        idTex = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
        string dir = Path.Combine(outputRoot, episode);
        Directory.CreateDirectory(Path.Combine(dir, "images"));
        Directory.CreateDirectory(Path.Combine(dir, "masks"));
        index = Directory.GetFiles(Path.Combine(dir, "images"), "*.png").Length;   // continue numbering
    }

    void Update()
    {
        if (Input.GetKeyDown(toggleKey))
        {
            recording = !recording;
            Debug.Log($"SurgSegCapture: recording {(recording ? "ON" : "OFF")} ({index} pairs so far)");
        }
        if (recording && Time.time >= next)
        {
            next = Time.time + interval;
            StartCoroutine(CaptureAtEndOfFrame());
        }
    }

    IEnumerator CaptureAtEndOfFrame()
    {
        yield return new WaitForEndOfFrame();
        Capture();
    }

    void OnGUI()
    {
        var style = new GUIStyle(GUI.skin.box) { fontSize = 14, richText = true, alignment = TextAnchor.MiddleCenter };
        string text = recording ? $"<color=#ff3030>● REC</color>  {index} frames" : $"F9 = record   ({index} saved)";
        GUI.Box(new Rect(Screen.width - 200, 8, 192, 26), text, style);
    }

    public void Capture()
    {
        RenderInto(rgbRT, rgbTex);                 // 1) what the trainee sees

        var swapped = new List<(Renderer r, Material[] mats)>();
        var cam = endoscopeCamera;
        var oldClear = cam.clearFlags; var oldBg = cam.backgroundColor;
        bool oldHdr = cam.allowHDR, oldMsaa = cam.allowMSAA;
        object urp = cam.GetComponent("UniversalAdditionalCameraData");
        var pPost = urp?.GetType().GetProperty("renderPostProcessing");
        var pAA = urp?.GetType().GetProperty("antialiasing");
        object oldPost = pPost?.GetValue(urp), oldAA = pAA?.GetValue(urp);
        try
        {
            // 2) label render: every renderer drawn flat with its class id
            foreach (var r in FindObjectsOfType<Renderer>())
            {
                if (!r.enabled || !r.gameObject.activeInHierarchy) continue;
                swapped.Add((r, r.sharedMaterials));
                var m = IdMaterial(LabelOf(r));
                var mats = new Material[Mathf.Max(1, r.sharedMaterials.Length)];
                for (int i = 0; i < mats.Length; i++) mats[i] = m;
                r.sharedMaterials = mats;
            }
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.black;        // empty space = background (0)
            cam.allowHDR = false;
            cam.allowMSAA = false;
            pPost?.SetValue(urp, false);              // URP: no tonemapping/bloom on ids
            if (pAA != null) pAA.SetValue(urp, System.Enum.ToObject(pAA.PropertyType, 0));
            RenderInto(idRT, idTex);
        }
        finally
        {
            foreach (var (r, mats) in swapped) r.sharedMaterials = mats;
            cam.clearFlags = oldClear; cam.backgroundColor = oldBg;
            cam.allowHDR = oldHdr; cam.allowMSAA = oldMsaa;
            if (pPost != null) pPost.SetValue(urp, oldPost);
            if (pAA != null) pAA.SetValue(urp, oldAA);
        }

        string dir = Path.Combine(outputRoot, episode);
        string name = $"{index:D6}.png";
        File.WriteAllBytes(Path.Combine(dir, "images", name), rgbTex.EncodeToPNG());
        File.WriteAllBytes(Path.Combine(dir, "masks", name), idTex.EncodeToPNG());
        index++;
    }

    void RenderInto(RenderTexture rt, Texture2D tex)
    {
        var prevTarget = endoscopeCamera.targetTexture;
        var prevActive = RenderTexture.active;
        endoscopeCamera.targetTexture = rt;
        endoscopeCamera.Render();
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();
        endoscopeCamera.targetTexture = prevTarget;
        RenderTexture.active = prevActive;
    }

    static int LabelOf(Renderer r)
    {
        var l = r.GetComponentInParent<SurgSegLabel>();
        return l != null ? (int)l.label : 0;
    }

    Material IdMaterial(int id)
    {
        if (!idMats.TryGetValue(id, out var m))
        {
            m = new Material(idShader);
            m.SetFloat("_SegId", id);
            idMats[id] = m;
        }
        return m;
    }

    void OnDestroy()
    {
        if (rgbRT) rgbRT.Release();
        if (idRT) idRT.Release();
        foreach (var m in idMats.Values) Destroy(m);
    }
}
