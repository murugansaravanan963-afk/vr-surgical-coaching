// World-locked overlay: re-projects the model's role colours from the captured view onto the
// surfaces seen in the current view. Used by SurgSegWorldLock (image effect on the camera).
Shader "Hidden/SurgSeg/Reproject"
{
    Properties { _MainTex ("", 2D) = "white" {} }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            sampler2D _RoleTex;
            sampler2D_float _CapDepth;
            float4x4 _CapView, _CapVP, _InvProj, _CamToWorld;
            float _FarClip, _DepthTol, _Opacity;

            fixed4 frag (v2f_img i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);
                float2 uv = i.uv;
                #if UNITY_UV_STARTS_AT_TOP
                if (_MainTex_TexelSize.y < 0) uv.y = 1 - uv.y;
                #endif
                float eye = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv));
                if (eye >= _FarClip * 0.999) return col;                 // background

                // surface point seen now -> world space
                float4 far = mul(_InvProj, float4(uv * 2 - 1, 1, 1));
                far.xyz /= far.w;
                float3 viewPos = far.xyz * (eye / -far.z);
                float4 world = mul(_CamToWorld, float4(viewPos, 1));

                // -> where it was in the frame the model saw
                float4 clip = mul(_CapVP, world);
                if (clip.w <= 0) return col;
                float2 cuv = clip.xy / clip.w * 0.5 + 0.5;
                if (any(cuv < 0) || any(cuv > 1)) return col;            // outside the captured view

                // only if that same surface was visible then (not hidden behind a tool)
                float capEye = -mul(_CapView, world).z;
                float stored = tex2D(_CapDepth, cuv).r;
                if (stored <= 0 || abs(capEye - stored) > max(_DepthTol, stored * 0.03)) return col;

                fixed4 o = tex2D(_RoleTex, cuv);
                if (o.b > 0.5 && o.r < 0.3) return col;                  // instrument: tools move on their own
                col.rgb = lerp(col.rgb, o.rgb, o.a * _Opacity);
                return col;
            }
            ENDCG
        }
    }
}
