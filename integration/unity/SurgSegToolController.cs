// Keyboard-steered instrument for the SurgSeg organ scene (a stand-in for the da Vinci PSM).
// The transform this sits on IS the instrument tip; the shaft always passes through `pivot`
// (the trocar / remote centre of motion), like a real laparoscopic or robotic instrument.
//   W / S  : up / down          A / D : left / right      (relative to the endoscope view)
//   E / Q  : push in / pull back                          hold Shift for faster motion
using UnityEngine;

public class SurgSegToolController : MonoBehaviour
{
    public Camera viewCamera;
    public Transform pivot;
    public float speed = 0.03f;              // metres per second

    void Update()
    {
        if (viewCamera == null) return;
        Transform c = viewCamera.transform;
        Vector3 d = Vector3.zero;
        if (Input.GetKey(KeyCode.W)) d += c.up;
        if (Input.GetKey(KeyCode.S)) d -= c.up;
        if (Input.GetKey(KeyCode.D)) d += c.right;
        if (Input.GetKey(KeyCode.A)) d -= c.right;
        if (Input.GetKey(KeyCode.E)) d += c.forward;
        if (Input.GetKey(KeyCode.Q)) d -= c.forward;
        float s = speed * (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift) ? 3f : 1f);
        transform.position += d * (s * Time.deltaTime);
        AimShaft();
    }

    public void AimShaft()
    {
        if (pivot == null) return;
        Vector3 back = pivot.position - transform.position;      // tool's local +Y runs up the shaft
        if (back.sqrMagnitude > 1e-8f) transform.rotation = Quaternion.FromToRotation(Vector3.up, back.normalized);
    }
}
