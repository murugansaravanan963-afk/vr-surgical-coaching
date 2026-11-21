// Writes the object's class id into the image (R=G=B=id/255), unlit, no blending.
// Used only by SurgSegCapture for the label-mask render. Works in Built-in and URP.
Shader "Hidden/SurgSegId"
{
    Properties { _SegId ("Segmentation id", Float) = 0 }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }
        Pass
        {
            ZWrite On
            ZTest LEqual
            Cull Off
            Blend Off
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            float _SegId;   // a float, not a Color: colours get gamma-converted, floats don't
            struct appdata { float4 vertex : POSITION; };
            struct v2f { float4 pos : SV_POSITION; };
            v2f vert (appdata v) { v2f o; o.pos = UnityObjectToClipPos(v.vertex); return o; }
            float4 frag (v2f i) : SV_Target { float c = _SegId / 255.0; return float4(c, c, c, 1); }
            ENDCG
        }
    }
}
