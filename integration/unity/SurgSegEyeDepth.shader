// Replacement shader: writes linear eye depth (metres) of every surface. SurgSegWorldLock
// renders the captured frame with it so the overlay can be re-projected later.
Shader "Hidden/SurgSeg/EyeDepth"
{
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        Pass
        {
            Cull Off ZWrite On ZTest LEqual
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct v2f { float4 pos : SV_POSITION; float eye : TEXCOORD0; };

            v2f vert (appdata_base v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.eye = -UnityObjectToViewPos(v.vertex).z;
                return o;
            }

            float4 frag (v2f i) : SV_Target { return float4(i.eye, 0, 0, 1); }
            ENDCG
        }
    }
}
