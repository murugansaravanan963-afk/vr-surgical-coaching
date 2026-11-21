// Tag a tissue / tool GameObject with its surgseg class. Applies to all child renderers
// unless a child has its own SurgSegLabel. Unlabelled objects count as background.
// The numbers MUST match configs/taxonomy.yaml (they are written straight into the masks).
using UnityEngine;

public enum SurgClass
{
    background = 0, instrument = 1, liver = 2, gallbladder = 3, stomach = 4,
    small_intestine = 5, colon = 6, pancreas = 7, spleen = 8, kidney = 9, ureter = 10,
    vesicular_glands = 11, artery = 12, vein = 13, duct = 14, fat = 15,
    connective_tissue = 16, abdominal_wall = 17, blood = 18, suture = 19,
    surgical_object = 20, uterus = 21,
}

public class SurgSegLabel : MonoBehaviour
{
    public SurgClass label = SurgClass.liver;
}
