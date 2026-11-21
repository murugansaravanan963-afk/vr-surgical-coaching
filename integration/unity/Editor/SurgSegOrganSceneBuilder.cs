// Menu: SurgSeg > Build Organ Scene
// Builds a new scene "Assets/Scenes/SurgSeg Organ Scene.unity" with:
//   - real anatomy (BodyParts3D, CT-derived): liver, gallbladder, stomach, duodenum,
//     common hepatic artery, plus generated cystic duct + cystic artery tubes
//   - an endoscope camera with a surgical light
//   - a keyboard-steered da Vinci needle-driver instrument (meshes from CRESSim if present)
//   - the SurgSeg overlay + model client (target/danger colours and tip warnings)
//   - the SurgSegCapture recorder (F9) for fine-tuning data
// Anatomy: BodyParts3D, (c) The Database Center for Life Science, CC BY-SA 2.1 Japan.
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

public static class SurgSegOrganSceneBuilder
{
    const string Root = "Assets/unity";
    const string AnatomyDir = Root + "/Anatomy";
    const string GenDir = AnatomyDir + "/Generated";
    const string ScenePath = "Assets/Scenes/SurgSeg Organ Scene.unity";
    const string ToolDir = "Assets/Models/PSM/Large Needle Driver";
    static readonly Vector3 Focus = new Vector3(0f, 1f, 0f);    // where the gallbladder neck is placed

    struct Organ
    {
        public string file; public SurgClass label; public Color color; public float gloss;
        public Organ(string f, SurgClass l, Color c, float g) { file = f; label = l; color = c; gloss = g; }
    }

    static readonly Organ[] Organs =
    {
        new Organ("liver", SurgClass.liver, new Color(0.36f, 0.08f, 0.07f), 0.6f),
        new Organ("gallbladder", SurgClass.gallbladder, new Color(0.40f, 0.55f, 0.30f), 0.75f),
        new Organ("stomach", SurgClass.stomach, new Color(0.80f, 0.52f, 0.47f), 0.55f),
        new Organ("duodenum", SurgClass.small_intestine, new Color(0.82f, 0.60f, 0.50f), 0.55f),
        new Organ("common_hepatic_artery", SurgClass.artery, new Color(0.78f, 0.14f, 0.14f), 0.7f),
    };

    [MenuItem("SurgSeg/Build Organ Scene")]
    public static void Build()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        Directory.CreateDirectory(GenDir);
        AssetDatabase.Refresh();
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        // ---------------- anatomy (BodyParts3D: millimetres; after Unity's OBJ import
        // local +x = patient right, +y = posterior, +z = superior) ----------------
        var anatomy = new GameObject("SurgSeg Anatomy");
        var bp = new GameObject("BodyParts3D (mm)");
        bp.transform.SetParent(anatomy.transform, false);
        bp.transform.localScale = Vector3.one * 0.001f;
        var parts = new Dictionary<string, GameObject>();
        foreach (var o in Organs)
        {
            string path = $"{AnatomyDir}/{o.file}.obj";
            var imp = AssetImporter.GetAtPath(path) as ModelImporter;
            if (imp == null) { Debug.LogError($"SurgSeg: missing {path}"); continue; }
            if (!imp.isReadable) { imp.isReadable = true; imp.SaveAndReimport(); }
            var model = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            var go = Object.Instantiate(model, bp.transform, false);
            go.name = o.file;
            Paint(go, SaveMaterial(o.file, o.color, o.gloss));
            AddColliders(go);   // so the user can click on the organ to mark it
            go.AddComponent<SurgSegLabel>().label = o.label;
            parts[o.file] = go;
        }
        if (!parts.ContainsKey("gallbladder") || !parts.ContainsKey("common_hepatic_artery"))
        {
            EditorUtility.DisplayDialog("SurgSeg", "Anatomy files missing in " + AnatomyDir, "OK");
            return;
        }

        // ---------------- cystic duct + cystic artery (not in BodyParts3D: generated) ----------------
        var gb = WorldVertices(parts["gallbladder"]);
        var cha = WorldVertices(parts["common_hepatic_artery"]);
        Vector3 gbCentre = Centroid(gb), chaCentre = Centroid(cha);
        Vector3 neck = Closest(gb, chaCentre);
        Vector3 chaNear = Closest(cha, neck);
        // Work out how the OBJ importer mapped BodyParts3D's axes (x = patient left,
        // y = posterior, z = superior) by comparing raw-file and imported gallbladder centres.
        Vector3 rawC = ObjBoundsCentre($"{AnatomyDir}/gallbladder.obj");
        Vector3 impC = LocalBoundsCentre(parts["gallbladder"], bp.transform);
        Vector3 superior = AxisDir(2, rawC, impC, bp.transform);
        Vector3 anterior = -AxisDir(1, rawC, impC, bp.transform);
        Vector3 patientLeft = AxisDir(0, rawC, impC, bp.transform);
        Vector3 inward = (gbCentre - neck).normalized;

        // Liver retraction, as in a real cholecystectomy: lift the liver ~30 deg about its
        // posterior-superior edge so the gallbladder neck and Calot's triangle are exposed.
        // (Ray-cast check on this anatomy: duct/artery visibility 0/12 -> 11/12.)
        if (parts.ContainsKey("liver"))
        {
            var liv = WorldVertices(parts["liver"]);
            Vector3 c = Centroid(liv);
            float maxPost = float.MinValue, maxSup = float.MinValue;
            Vector3 lowFront = c; float best = float.MaxValue;
            foreach (var v in liv)
            {
                maxPost = Mathf.Max(maxPost, Vector3.Dot(v, -anterior));
                maxSup = Mathf.Max(maxSup, Vector3.Dot(v, superior));
                float score = Vector3.Dot(v, superior) - Vector3.Dot(v, anterior);   // low and front
                if (score < best) { best = score; lowFront = v; }
            }
            Vector3 hinge = c + (-anterior) * (maxPost - Vector3.Dot(c, -anterior)) + superior * (maxSup - Vector3.Dot(c, superior));
            // pick the rotation sign that lifts the anterior-inferior edge upwards
            Vector3 up30 = Quaternion.AngleAxis(30f, patientLeft) * (lowFront - hinge) + hinge;
            float sign = Vector3.Dot(up30 - lowFront, superior) > 0f ? 1f : -1f;
            parts["liver"].transform.RotateAround(hinge, patientLeft, 30f * sign);
        }

        const float fwd = 0.018f;   // route duct/artery on the exposed (anterior) side
        Vector3 d0 = neck + inward * 0.002f;
        Vector3 d2 = Vector3.Lerp(neck, chaNear, 0.9f) - superior * 0.012f + anterior * (fwd * 0.5f);
        Vector3 d1 = (d0 + d2) * 0.5f + anterior * fwd;
        AddTube(anatomy, "cystic_duct", SurgClass.duct, new[] { d0, d1, d2 }, 0.0022f,
                SaveMaterial("cystic_duct", new Color(0.80f, 0.82f, 0.58f), 0.7f));

        Vector3 a0 = chaNear + anterior * (fwd * 0.3f);
        Vector3 a2 = Closest(gb, Vector3.Lerp(neck, gbCentre, 0.35f));
        Vector3 a1 = (a0 + a2) * 0.5f + anterior * (fwd + 0.002f) + superior * 0.004f;
        AddTube(anatomy, "cystic_artery", SurgClass.artery, new[] { a0, a1, a2 }, 0.0013f,
                SaveMaterial("cystic_artery", new Color(0.80f, 0.12f, 0.12f), 0.7f));

        // retroperitoneal fat backdrop behind the organs
        var back = GameObject.CreatePrimitive(PrimitiveType.Quad);
        Object.DestroyImmediate(back.GetComponent<Collider>());   // marks pin to organs, never to the backdrop
        back.name = "fat_backdrop";
        back.transform.SetParent(anatomy.transform, false);
        back.transform.position = neck - anterior * 0.09f;
        back.transform.rotation = Quaternion.LookRotation(-anterior, superior);
        back.transform.localScale = Vector3.one * 0.6f;
        back.GetComponent<MeshRenderer>().sharedMaterial = SaveMaterial("fat", new Color(0.78f, 0.62f, 0.38f), 0.45f);
        back.AddComponent<SurgSegLabel>().label = SurgClass.fat;

        // orient: superior -> up, anterior -> +Z (towards the camera); gallbladder neck at Focus
        Quaternion r = Quaternion.Inverse(Quaternion.LookRotation(anterior, superior));
        anatomy.transform.rotation = r;
        anatomy.transform.position = Focus - r * neck;
        // Viewed from the front, the patient's left must appear on the viewer's right
        // (camera right = world -X). Mirror about the focus point if the import flipped it.
        if ((r * patientLeft).x > 0f)
        {
            var mirror = new GameObject("SurgSeg Anatomy (mirrored to anatomical view)");
            mirror.transform.position = Focus;
            anatomy.transform.SetParent(mirror.transform, true);
            mirror.transform.localScale = new Vector3(-1f, 1f, 1f);
        }

        // ---------------- endoscope camera + light ----------------
        var camGo = new GameObject("SurgSeg Endoscope");
        camGo.tag = "MainCamera";
        var cam = camGo.AddComponent<Camera>();
        cam.fieldOfView = 70f; cam.nearClipPlane = 0.003f; cam.farClipPlane = 5f;
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.10f, 0.03f, 0.03f);
        camGo.transform.position = Focus + new Vector3(0f, -0.07f, 0.17f);   // ~18 cm: whole hepatocystic region in view
        camGo.transform.LookAt(Focus + new Vector3(0f, 0.01f, 0f), Vector3.up);
        var orbit = camGo.AddComponent<SurgSegCameraOrbit>();   // arrows / PageUp-Down / R = random view
        orbit.target = Focus + new Vector3(0f, 0.01f, 0f);
        var spot = new GameObject("Endoscope Light").AddComponent<Light>();
        spot.transform.SetParent(camGo.transform, false);
        spot.type = LightType.Spot; spot.range = 1.5f; spot.spotAngle = 100f; spot.intensity = 1.2f;
        spot.color = new Color(1f, 0.95f, 0.88f); spot.shadows = LightShadows.Soft;
        var fill = new GameObject("Fill Light").AddComponent<Light>();
        fill.type = LightType.Directional; fill.intensity = 0.25f;
        fill.transform.rotation = Quaternion.LookRotation(new Vector3(0.2f, -0.5f, -1f));
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.22f, 0.17f, 0.16f);

        // ---------------- da Vinci instrument (keyboard) ----------------
        var pivot = new GameObject("Trocar (pivot)").transform;
        pivot.position = camGo.transform.position + camGo.transform.right * 0.09f + camGo.transform.up * 0.02f;
        var tool = BuildInstrument();
        tool.transform.position = Focus + camGo.transform.right * 0.03f + Vector3.forward * 0.025f;
        var ctl = tool.AddComponent<SurgSegToolController>();
        ctl.viewCamera = cam; ctl.pivot = pivot; ctl.AimShaft();

        // ---------------- overlay + model client + recorder ----------------
        var canvasGo = new GameObject("SurgSeg Overlay", typeof(Canvas), typeof(CanvasScaler));
        canvasGo.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
        var imgGo = new GameObject("RoleOverlay", typeof(RectTransform), typeof(RawImage));
        imgGo.transform.SetParent(canvasGo.transform, false);
        var rect = imgGo.GetComponent<RectTransform>();
        rect.anchorMin = Vector2.zero; rect.anchorMax = Vector2.one; rect.offsetMin = Vector2.zero; rect.offsetMax = Vector2.zero;
        var raw = imgGo.GetComponent<RawImage>();
        raw.raycastTarget = false;

        var seg = new GameObject("SurgSeg");
        var client = seg.AddComponent<SurgSegClient>();
        client.endoscopeCamera = cam;
        client.toolTips = new[] { tool.transform };
        client.roleOverlay = raw;
        client.procedure = "cholecystectomy";
        client.mode = "auto";
        var cap = seg.AddComponent<SurgSegCapture>();
        cap.endoscopeCamera = cam;
        cap.idShader = AssetDatabase.LoadAssetAtPath<Shader>(Root + "/SurgSegId.shader");
        cap.episode = "organ_ep01";

        Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
        EditorSceneManager.SaveScene(scene, ScenePath);
        Selection.activeGameObject = seg;
        EditorUtility.DisplayDialog("SurgSeg",
            "Organ scene built and saved:\n" + ScenePath +
            "\n\n1. Make sure the model server is running (D:\\surgseg\\start_server.bat)" +
            "\n2. Press Play" +
            "\n3. Click the Game view. Instrument: W/S up-down, A/D left-right, E/Q in-out (Shift = faster)" +
            "\n   Camera: arrow keys orbit, PageUp/PageDown zoom, R = random view" +
            "\n4. F9 records training frames for fine-tuning", "OK");
    }

    // ---------------------------------------------------------------- helpers
    static GameObject BuildInstrument()
    {
        // Tool root sits at the jaw tip; local +Y runs back up the shaft.
        var tool = new GameObject("PSM1");
        tool.AddComponent<SurgSegLabel>().label = SurgClass.instrument;
        var metal = SaveMaterial("instrument_metal", new Color(0.62f, 0.63f, 0.65f), 0.85f);
        var shaftMat = SaveMaterial("instrument_shaft", new Color(0.12f, 0.12f, 0.13f), 0.6f);
        bool ok = AddToolPart(tool, "psm_tool_large_needle_driver_gripper_1", new Vector3(0f, 0.0106f, 0f), Quaternion.Euler(0f, 0f, -8f), metal)
                & AddToolPart(tool, "psm_tool_large_needle_driver_gripper_2", new Vector3(0f, 0.0106f, 0f), Quaternion.Euler(0f, 0f, 8f), metal)
                & AddToolPart(tool, "psm_tool_large_needle_driver_pitch", new Vector3(0f, 0.0197f, 0f), Quaternion.identity, metal)
                & AddToolPart(tool, "psm_tool_large_needle_driver_roll", new Vector3(0f, 0.2067f, 0f), Quaternion.identity, shaftMat);
        if (!ok)   // CRESSim meshes not found: simple stand-in
        {
            foreach (Transform c in tool.transform) Object.DestroyImmediate(c.gameObject);
            var shaft = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            Object.DestroyImmediate(shaft.GetComponent<Collider>());
            shaft.transform.SetParent(tool.transform, false);
            shaft.transform.localPosition = new Vector3(0f, 0.2f, 0f);
            shaft.transform.localScale = new Vector3(0.008f, 0.19f, 0.008f);
            shaft.GetComponent<MeshRenderer>().sharedMaterial = shaftMat;
            var jaw = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Object.DestroyImmediate(jaw.GetComponent<Collider>());
            jaw.transform.SetParent(tool.transform, false);
            jaw.transform.localPosition = new Vector3(0f, 0.006f, 0f);
            jaw.transform.localScale = new Vector3(0.004f, 0.012f, 0.003f);
            jaw.GetComponent<MeshRenderer>().sharedMaterial = metal;
        }
        return tool;
    }

    static bool AddToolPart(GameObject tool, string file, Vector3 pos, Quaternion rot, Material mat)
    {
        var model = AssetDatabase.LoadAssetAtPath<GameObject>($"{ToolDir}/{file}.obj");
        if (model == null) return false;
        var go = Object.Instantiate(model, tool.transform, false);
        go.name = file;
        go.transform.localPosition = pos;
        go.transform.localRotation = rot;
        Paint(go, mat);
        return true;
    }

    static void AddColliders(GameObject go)
    {
        foreach (var mf in go.GetComponentsInChildren<MeshFilter>())
            if (mf.sharedMesh != null && mf.GetComponent<Collider>() == null)
                mf.gameObject.AddComponent<MeshCollider>().sharedMesh = mf.sharedMesh;
    }

    static void Paint(GameObject go, Material m)
    {
        foreach (var r in go.GetComponentsInChildren<MeshRenderer>())
        {
            var mats = new Material[Mathf.Max(1, r.sharedMaterials.Length)];
            for (int i = 0; i < mats.Length; i++) mats[i] = m;
            r.sharedMaterials = mats;
        }
    }

    static Material SaveMaterial(string name, Color color, float gloss)
    {
        string path = $"{GenDir}/{name}.mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m == null)
        {
            m = new Material(Shader.Find("Standard"));
            AssetDatabase.CreateAsset(m, path);
        }
        m.color = color;
        m.SetFloat("_Glossiness", gloss);
        m.SetFloat("_Metallic", name.StartsWith("instrument") ? 0.8f : 0f);
        EditorUtility.SetDirty(m);
        return m;
    }

    static Vector3 ObjBoundsCentre(string assetPath)
    {
        Vector3 mn = Vector3.one * float.MaxValue, mx = Vector3.one * float.MinValue;
        foreach (var line in File.ReadLines(assetPath))
        {
            if (!line.StartsWith("v ")) continue;
            var t = line.Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries);
            var v = new Vector3(float.Parse(t[1], System.Globalization.CultureInfo.InvariantCulture),
                                float.Parse(t[2], System.Globalization.CultureInfo.InvariantCulture),
                                float.Parse(t[3], System.Globalization.CultureInfo.InvariantCulture));
            mn = Vector3.Min(mn, v); mx = Vector3.Max(mx, v);
        }
        return (mn + mx) * 0.5f;
    }

    static Vector3 LocalBoundsCentre(GameObject go, Transform space)
    {
        Vector3 mn = Vector3.one * float.MaxValue, mx = Vector3.one * float.MinValue;
        foreach (var w in WorldVertices(go))
        {
            var v = space.InverseTransformPoint(w);
            mn = Vector3.Min(mn, v); mx = Vector3.Max(mx, v);
        }
        return (mn + mx) * 0.5f;
    }

    // World direction of raw-file axis `rawAxis`, found by matching |centre| values per axis.
    static Vector3 AxisDir(int rawAxis, Vector3 rawC, Vector3 impC, Transform space)
    {
        int best = 0; float bestErr = float.MaxValue;
        for (int j = 0; j < 3; j++)
        {
            float err = Mathf.Abs(Mathf.Abs(impC[j]) - Mathf.Abs(rawC[rawAxis]));
            if (err < bestErr) { bestErr = err; best = j; }
        }
        Vector3 local = Vector3.zero;
        local[best] = Mathf.Sign(impC[best]) * Mathf.Sign(rawC[rawAxis]);
        return space.TransformDirection(local).normalized;
    }

    static List<Vector3> WorldVertices(GameObject go)
    {
        var list = new List<Vector3>();
        foreach (var mf in go.GetComponentsInChildren<MeshFilter>())
        {
            if (mf.sharedMesh == null) continue;
            var m = mf.transform.localToWorldMatrix;
            foreach (var v in mf.sharedMesh.vertices) list.Add(m.MultiplyPoint3x4(v));
        }
        return list;
    }

    static Vector3 Centroid(List<Vector3> pts)
    {
        Vector3 s = Vector3.zero;
        foreach (var p in pts) s += p;
        return pts.Count > 0 ? s / pts.Count : s;
    }

    static Vector3 Closest(List<Vector3> pts, Vector3 q)
    {
        Vector3 best = q; float bd = float.MaxValue;
        foreach (var p in pts) { float d = (p - q).sqrMagnitude; if (d < bd) { bd = d; best = p; } }
        return best;
    }

    static void AddTube(GameObject parent, string name, SurgClass label, Vector3[] ctrl, float radius, Material mat)
    {
        // quadratic Bezier through ctrl[0..2], swept circle, written in the parent's local space
        const int seg = 32, sides = 14;
        var verts = new List<Vector3>(); var tris = new List<int>();
        Vector3 Bez(float t) => (1 - t) * (1 - t) * ctrl[0] + 2 * (1 - t) * t * ctrl[1] + t * t * ctrl[2];
        Vector3 prevN = Vector3.zero;
        for (int i = 0; i <= seg; i++)
        {
            float t = i / (float)seg;
            Vector3 p = Bez(t);
            Vector3 tan = (Bez(Mathf.Min(1f, t + 0.01f)) - Bez(Mathf.Max(0f, t - 0.01f))).normalized;
            Vector3 n = i == 0 ? Vector3.Cross(tan, Mathf.Abs(tan.y) < 0.9f ? Vector3.up : Vector3.right).normalized
                               : Vector3.Cross(Vector3.Cross(tan, prevN), tan).normalized;   // parallel transport
            prevN = n;
            Vector3 b = Vector3.Cross(tan, n);
            float taper = 1f - 0.25f * Mathf.Abs(2f * t - 1f);
            for (int k = 0; k < sides; k++)
            {
                float a = 2f * Mathf.PI * k / sides;
                verts.Add(parent.transform.InverseTransformPoint(p + (n * Mathf.Cos(a) + b * Mathf.Sin(a)) * radius * taper));
            }
        }
        for (int i = 0; i < seg; i++)
            for (int k = 0; k < sides; k++)
            {
                int a0 = i * sides + k, a1 = i * sides + (k + 1) % sides, b0 = a0 + sides, b1 = a1 + sides;
                tris.AddRange(new[] { a0, b0, a1, a1, b0, b1 });
            }
        var mesh = new Mesh { name = name };
        mesh.SetVertices(verts); mesh.SetTriangles(tris, 0);
        mesh.RecalculateNormals(); mesh.RecalculateBounds();
        string path = $"{GenDir}/{name}.asset";
        AssetDatabase.DeleteAsset(path);
        AssetDatabase.CreateAsset(mesh, path);

        var go = new GameObject(name, typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.SetParent(parent.transform, false);
        go.GetComponent<MeshFilter>().sharedMesh = mesh;
        go.GetComponent<MeshRenderer>().sharedMaterial = mat;
        go.AddComponent<MeshCollider>().sharedMesh = mesh;
        go.AddComponent<SurgSegLabel>().label = label;
    }
}
