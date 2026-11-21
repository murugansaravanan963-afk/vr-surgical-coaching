// Endoscope camera controls for the SurgSeg organ scene (vary the view while recording).
//   Arrow keys : orbit around the target     PageUp / PageDown (or + / -) : zoom in / out
//   R          : random view (handy while recording training frames with F9)
using UnityEngine;

public class SurgSegCameraOrbit : MonoBehaviour
{
    public Vector3 target = new Vector3(0f, 1f, 0f);
    public float minDistance = 0.08f, maxDistance = 0.30f;
    public float orbitSpeed = 40f;          // degrees per second
    public float zoomSpeed = 0.08f;         // metres per second

    float yaw, pitch, distance;

    void Start()
    {
        Vector3 off = transform.position - target;
        distance = off.magnitude;
        yaw = Mathf.Atan2(off.x, off.z) * Mathf.Rad2Deg;
        pitch = Mathf.Asin(Mathf.Clamp(off.y / Mathf.Max(distance, 1e-4f), -1f, 1f)) * Mathf.Rad2Deg;
    }

    void Update()
    {
        float dt = Time.deltaTime;
        if (Input.GetKey(KeyCode.LeftArrow)) yaw -= orbitSpeed * dt;
        if (Input.GetKey(KeyCode.RightArrow)) yaw += orbitSpeed * dt;
        if (Input.GetKey(KeyCode.UpArrow)) pitch += orbitSpeed * dt;
        if (Input.GetKey(KeyCode.DownArrow)) pitch -= orbitSpeed * dt;
        if (Input.GetKey(KeyCode.PageUp) || Input.GetKey(KeyCode.Equals) || Input.GetKey(KeyCode.KeypadPlus)) distance -= zoomSpeed * dt;
        if (Input.GetKey(KeyCode.PageDown) || Input.GetKey(KeyCode.Minus) || Input.GetKey(KeyCode.KeypadMinus)) distance += zoomSpeed * dt;
        if (Input.GetKeyDown(KeyCode.R))
        {
            yaw = Random.Range(-45f, 45f);
            pitch = Random.Range(-45f, 10f);
            distance = Random.Range(minDistance, maxDistance);
        }
        yaw = Mathf.Clamp(yaw, -70f, 70f);          // stay on the front (anterior) side of the anatomy
        pitch = Mathf.Clamp(pitch, -60f, 30f);
        distance = Mathf.Clamp(distance, minDistance, maxDistance);
        Quaternion q = Quaternion.Euler(-pitch, yaw, 0f);
        transform.position = target + q * (Vector3.forward * distance);
        transform.LookAt(target, Vector3.up);
    }
}
