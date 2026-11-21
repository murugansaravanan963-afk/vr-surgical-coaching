// Unity client for the surgseg model server (python -m surgseg.runtime.server).
//
// Sends the endoscope camera image + instrument-tip positions to the server and shows:
//   - the colour overlay (green = target, red = danger, faint blue = instruments)
//   - a heads-up panel with each tip's safety level (OK / WARNING / VIOLATION) and distance
// NetMQ.dll + AsyncIO.dll must be in the project (they ship in this folder's Plugins/).
//
// Coordinate conventions (must match the server):
//   u, v  : normalised image coords, origin TOP-LEFT  ->  u = vp.x, v = 1 - vp.y
//   xyz   : camera frame, OpenCV convention (x right, y down, z forward), metres
using System.Collections.Generic;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using UnityEngine.UI;

public class SurgSegClient : MonoBehaviour
{
    [Header("Connection")]
    public string address = "tcp://127.0.0.1:5555";
    public string mode = "auto";                 // "auto" | "manual"
    public string procedure = "cholecystectomy";
    public float replyTimeoutSeconds = 5f;       // slow GPUs can take ~1 s per frame

    [Header("Scene")]
    public Camera endoscopeCamera;
    public Transform[] toolTips;                 // e.g. PSM1 / PSM2 jaw tips
    public RawImage roleOverlay;                 // full-screen RawImage on a Screen Space - Overlay canvas
    public GameObject labelPrefab;               // optional: anchored-to-tissue labels (needs a TMP_Text)
    public int width = 800, height = 512;
    public bool matchScreenAspect = true;        // keeps the overlay aligned with the full-screen view
    public bool showHud = true;

    [Header("User marking (manual mode; also works as overrides in auto mode)")]
    public bool mousePicking = true;             // left-click = TARGET, right-click = DANGER
    public KeyCode toggleModeKey = KeyCode.M;    // switch manual <-> auto
    public KeyCode clearMarksKey = KeyCode.C;
    public Transform pointer;                    // optional VR controller: trigger marks DANGER
    public float pointerRange = 2f;
    [Header("Overlay locked to the tissue while the view moves")]
    public bool worldLocked = true;              // false = old flat screen overlay (lags behind camera motion)
    public KeyCode toggleLockKey = KeyCode.L;

    class Mark { public Transform on; public Vector3 local; public Vector2 screenUv; public string role; }
    readonly List<Mark> marks = new List<Mark>();

    public System.Action<TipStatus> OnTipStatus; // hook up audio/haptics/visual alerts

    // --- JSON shapes (JsonUtility) ---
    [System.Serializable] public class TipStatus { public string id; public string level; public float dist_px; public float dist_mm; public string nearest_danger; public bool on_target; }
    [System.Serializable] public class Region { public string role; public string @class; public float[] anchor_uv; public float area_frac; public float mean_conf; }
    [System.Serializable] public class Timing { public float model; public float total; public float frame_age; }
    [System.Serializable] public class InferReply { public bool ok; public string reason; public int frame_id; public TipStatus[] tips; public Region[] regions; public Timing timing_ms; }

    RenderTexture rt; Texture2D readback, overlayTex;
    Thread worker; volatile bool running; volatile string status = "connecting to model server...";
    readonly object gate = new object();
    byte[] pendingFrame; string pendingHeader; readonly Queue<string> pendingCommands = new Queue<string>();
    string lastReply; byte[] lastRoleMap;
    readonly List<GameObject> labels = new List<GameObject>();
    TipStatus[] hudTips = new TipStatus[0];
    float lastReplyTime = -1f, replyInterval;
    int frameId;
    volatile bool inFlight;                      // a frame is at the model; capture the next only when it returns
    int inFlightFrame = -1, inFlightSlot = -1;
    SurgSegWorldLock worldLock;

    void Start()
    {
        if (matchScreenAspect && Screen.width > 0)
            height = Mathf.Max(32, Mathf.RoundToInt(width * (float)Screen.height / Screen.width));
        rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        readback = new Texture2D(width, height, TextureFormat.RGB24, false);
        overlayTex = new Texture2D(2, 2);
        if (roleOverlay != null) { roleOverlay.enabled = false; roleOverlay.raycastTarget = false; }   // hidden until the first result
        if (endoscopeCamera != null)
        {
            worldLock = endoscopeCamera.GetComponent<SurgSegWorldLock>();
            if (worldLock == null) worldLock = endoscopeCamera.gameObject.AddComponent<SurgSegWorldLock>();
            worldLock.active = worldLocked;
        }
        running = true;
        worker = new Thread(Worker) { IsBackground = true };
        worker.Start();
        Command($"{{\"cmd\":\"set_mode\",\"mode\":\"{mode}\",\"procedure\":\"{procedure}\"}}");
    }

    void LateUpdate()
    {
        if (endoscopeCamera == null) return;
        HandleMarkingInput();

        // 0) apply the latest result first, so its capture slot is on screen before we reuse the other
        string reply; byte[] map;
        lock (gate) { reply = lastReply; map = lastRoleMap; lastReply = null; lastRoleMap = null; }
        if (reply != null) Apply(JsonUtility.FromJson<InferReply>(reply), map);

        // send a new frame only when the model is free: the newest view goes out, and we keep
        // its depth + camera pose so the result can be locked onto the tissue when it returns
        bool free;
        lock (gate) free = !inFlight && pendingFrame == null;
        if (!free) return;

        // 1) grab the endoscope view (synchronous readback for clarity; use
        //    AsyncGPUReadback in production to avoid a GPU stall)
        SurgSegWorldLock.capturing = true;       // keep our own overlay out of the model's input
        var prev = endoscopeCamera.targetTexture;
        endoscopeCamera.targetTexture = rt;
        endoscopeCamera.Render();
        endoscopeCamera.targetTexture = prev;
        SurgSegWorldLock.capturing = false;
        RenderTexture.active = rt;
        readback.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        RenderTexture.active = null;
        int slot = worldLock != null ? worldLock.CaptureDepth(width, height) : -1;

        // 2) project tool tips (the sim knows them exactly - no need to segment them)
        var tips = new List<string>();
        if (toolTips != null)
            foreach (var t in toolTips)
            {
                if (t == null) continue;
                Vector3 vp = endoscopeCamera.WorldToViewportPoint(t.position);
                Vector3 c = endoscopeCamera.worldToCameraMatrix.MultiplyPoint(t.position);
                tips.Add(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                    "{{\"id\":\"{0}\",\"u\":{1},\"v\":{2},\"xyz\":[{3},{4},{5}]}}",
                    t.name, vp.x, 1f - vp.y, c.x, -c.y, -c.z));
            }
        // 3) user marks: mouse (or VR pointer) picks a point on the tissue; it is pinned to
        //    that surface point and re-projected every frame, so it follows camera motion.
        var anchors = new List<string>();
        foreach (var m in marks)
        {
            Vector2 uv = m.screenUv;
            if (m.on != null)
            {
                Vector3 vp = endoscopeCamera.WorldToViewportPoint(m.on.TransformPoint(m.local));
                uv = vp.z > 0 ? new Vector2(vp.x, 1f - vp.y) : new Vector2(-1f, -1f);   // behind camera -> skipped
            }
            anchors.Add(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{{\"u\":{0},\"v\":{1},\"role\":\"{2}\"}}", uv.x, uv.y, m.role));
        }

        string header = string.Format(System.Globalization.CultureInfo.InvariantCulture,
            "{{\"cmd\":\"infer\",\"frame_id\":{0},\"timestamp_ms\":{1},\"encoding\":\"raw_rgb8\"," +
            "\"width\":{2},\"height\":{3},\"flip_y\":true,\"tips\":[{4}],\"anchors\":[{5}],\"return\":[\"role_rgba\"]}}",
            frameId++, System.DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(), width, height,
            string.Join(",", tips), string.Join(",", anchors));
        inFlightFrame = frameId - 1;
        inFlightSlot = slot;
        lock (gate) { pendingFrame = readback.GetRawTextureData(); pendingHeader = header; inFlight = true; }
    }

    void HandleMarkingInput()
    {
        if (Input.GetKeyDown(toggleModeKey))
        {
            mode = mode == "auto" ? "manual" : "auto";
            SetMode(mode, procedure);
        }
        if (Input.GetKeyDown(clearMarksKey)) marks.Clear();
        if (Input.GetKeyDown(toggleLockKey))
        {
            worldLocked = !worldLocked;
            if (worldLock != null) worldLock.active = worldLocked;
            if (roleOverlay != null) roleOverlay.enabled = !worldLocked && roleOverlay.texture != null;
        }
        if (mousePicking)
        {
            if (Input.GetMouseButtonDown(0)) AddMark(endoscopeCamera.ScreenPointToRay(Input.mousePosition), Input.mousePosition, "target");
            if (Input.GetMouseButtonDown(1)) AddMark(endoscopeCamera.ScreenPointToRay(Input.mousePosition), Input.mousePosition, "danger");
        }
        if (pointer != null && Input.GetButtonDown("Fire1"))
            AddMark(new Ray(pointer.position, pointer.forward), endoscopeCamera.WorldToScreenPoint(pointer.position + pointer.forward * pointerRange), "danger");
    }

    void AddMark(Ray ray, Vector3 screenPos, string role)
    {
        var m = new Mark { role = role };
        if (Physics.Raycast(ray, out var hit, 10f))
        {   // pinned to the 3D surface: follows the tissue when the camera moves
            m.on = hit.transform;
            m.local = hit.transform.InverseTransformPoint(hit.point);
        }
        else
        {   // no collider in this simulator: fall back to a fixed screen position
            m.screenUv = new Vector2(screenPos.x / Screen.width, 1f - screenPos.y / Screen.height);
        }
        marks.Add(m);
    }

    public void Select(float u, float v, string role, string scope) =>
        Command(string.Format(System.Globalization.CultureInfo.InvariantCulture,
            "{{\"cmd\":\"select\",\"u\":{0},\"v\":{1},\"role\":\"{2}\",\"scope\":\"{3}\"}}", u, v, role, scope));

    public void SetMode(string newMode, string proc, string step = null) =>
        Command($"{{\"cmd\":\"set_mode\",\"mode\":\"{newMode}\",\"procedure\":\"{proc}\"" +
                (step != null ? $",\"step\":\"{step}\"}}" : "}"));

    void Command(string json) { lock (gate) pendingCommands.Enqueue(json); }

    void Apply(InferReply r, byte[] roleMapPng)
    {
        if (r == null || !r.ok) { status = "model server error: " + (r != null ? r.reason : "bad reply"); return; }
        if (lastReplyTime > 0) replyInterval = Time.realtimeSinceStartup - lastReplyTime;
        lastReplyTime = Time.realtimeSinceStartup;
        status = "model connected";
        if (roleMapPng != null)
        {
            overlayTex.LoadImage(roleMapPng);  // transparent PNG: green = target, red = danger
            if (worldLock != null && r.frame_id == inFlightFrame && inFlightSlot >= 0)
                worldLock.Show(inFlightSlot, overlayTex);   // lock it onto the view it was computed from
            if (roleOverlay != null)
            {
                roleOverlay.texture = overlayTex;
                roleOverlay.enabled = !worldLocked || worldLock == null;
            }
        }
        hudTips = r.tips ?? new TipStatus[0];
        foreach (var t in hudTips) OnTipStatus?.Invoke(t);

        // labels pinned to tissue: ray through the anchor pixel, place label at the hit
        foreach (var g in labels) Destroy(g);
        labels.Clear();
        if (labelPrefab == null || r.regions == null) return;
        foreach (var reg in r.regions)
        {
            var ray = endoscopeCamera.ViewportPointToRay(new Vector3(reg.anchor_uv[0], 1f - reg.anchor_uv[1], 0));
            if (!Physics.Raycast(ray, out var hit, 1f)) continue;
            var g = Instantiate(labelPrefab, hit.point, Quaternion.LookRotation(hit.point - endoscopeCamera.transform.position));
            var text = g.GetComponentInChildren<TextMesh>();
            if (text) text.text = $"{reg.role.ToUpper()}: {reg.@class}";
            labels.Add(g);
        }
    }

    void OnGUI()
    {
        if (!showHud) return;
        var box = new GUIStyle(GUI.skin.box) { fontSize = 14, alignment = TextAnchor.UpperLeft, richText = true };
        var lines = new List<string> { $"<b>SurgSeg</b>  {status}" + (replyInterval > 0 ? $"  ({1f / replyInterval:0.0} updates/s)" : "") };
        foreach (var t in hudTips)
        {
            string col = t.level == "violation" ? "#ff3030" : t.level == "warning" ? "#ffb000" : t.level == "stale" ? "#a0a0a0" : "#40e040";
            string dist = t.dist_mm > 0 ? $"{t.dist_mm:0.0} mm" : t.dist_px > 0 ? $"{t.dist_px:0} px" : "-";
            string near = string.IsNullOrEmpty(t.nearest_danger) ? "" : $" from {t.nearest_danger}";
            string tgt = t.on_target ? "  [on target]" : "";
            lines.Add($"{t.id}: <color={col}><b>{t.level.ToUpper()}</b></color>  {dist}{near}{tgt}");
        }
        int nt = 0, nd = 0;
        foreach (var m in marks) { if (m.role == "target") nt++; else nd++; }
        lines.Add($"Mode: <b>{mode.ToUpper()}</b> ({toggleModeKey} to switch)   marks: {nt} target, {nd} danger");
        lines.Add($"Overlay: <b>{(worldLocked ? "LOCKED TO TISSUE" : "FLAT ON SCREEN (lags)")}</b> ({toggleLockKey} to switch)");
        lines.Add("<size=11>L-click = mark TARGET   R-click = mark DANGER   C = clear marks</size>");
        lines.Add("<size=11>green = target   red = danger   blue = instrument</size>");
        GUI.Box(new Rect(8, 8, 430, 12 + 19 * lines.Count), string.Join("\n", lines), box);

        // draw the user's marks where they currently project on screen
        var dot = new GUIStyle(GUI.skin.label) { fontSize = 22, alignment = TextAnchor.MiddleCenter, richText = true };
        foreach (var m in marks)
        {
            Vector2 uv = m.screenUv;
            if (m.on != null)
            {
                Vector3 vp = endoscopeCamera.WorldToViewportPoint(m.on.TransformPoint(m.local));
                if (vp.z <= 0) continue;
                uv = new Vector2(vp.x, 1f - vp.y);
            }
            string c = m.role == "target" ? "#30ff30" : "#ff3030";
            GUI.Label(new Rect(uv.x * Screen.width - 15, uv.y * Screen.height - 15, 30, 30), $"<color={c}>◉</color>", dot);
        }
    }

    void Worker()
    {
        AsyncIO.ForceDotNet.Force();
        while (running)
        {
            using (var sock = new RequestSocket())
            {
                sock.Connect(address);
                bool healthy = true;
                while (running && healthy)
                {
                    string cmd = null, header = null; byte[] frame = null;
                    lock (gate)
                    {
                        if (pendingCommands.Count > 0) cmd = pendingCommands.Dequeue();
                        else if (pendingFrame != null) { header = pendingHeader; frame = pendingFrame; pendingFrame = null; }
                    }
                    if (cmd == null && header == null) { Thread.Sleep(1); continue; }
                    var msg = new NetMQMessage();
                    msg.Append(cmd ?? header);
                    if (frame != null) msg.Append(frame);
                    sock.SendMultipartMessage(msg);
                    var rep = new NetMQMessage();
                    if (!sock.TryReceiveMultipartMessage(System.TimeSpan.FromSeconds(replyTimeoutSeconds), ref rep))
                    {
                        // A REQ socket is unusable after a missed reply: rebuild it and retry.
                        status = "model server not answering - start D:\\surgseg\\start_server.bat";
                        if (cmd != null) Command(cmd);
                        if (header != null) inFlight = false;
                        healthy = false;
                        continue;
                    }
                    if (header != null)
                        lock (gate) { lastReply = rep[0].ConvertToString(); lastRoleMap = rep.FrameCount > 1 ? rep[1].ToByteArray() : null; inFlight = false; }
                }
            }
            if (running) Thread.Sleep(500);
        }
        NetMQConfig.Cleanup(false);
    }

    void OnDestroy() { running = false; worker?.Join(500); }
}
