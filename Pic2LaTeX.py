import os
import io
import re
import base64
import tempfile
import shutil
import streamlit as st
from PIL import Image
from onnxruntime import InferenceSession
from models.paddleocr import predict_det, predict_rec, utility
from models.utils import mix_inference
from models.det_model.inference import PredictConfig
from models.ocr_model.model.Pic2LaTeX import Pic2LaTeX
from models.ocr_model.utils.inference import inference as latex_recognition
from models.ocr_model.utils.to_katex import to_katex
import streamlit.config
import streamlit.web.bootstrap

st.set_page_config(page_title="Pic2LaTeX",page_icon="🧮")

current_dir = os.path.dirname(os.path.abspath(__file__))
fire_path = os.path.join(current_dir, "assets", "fire.svg")

# 在Streamlit中会因为安全限制而无法正常显示本地图像。因此将SVG文件内容读取后转换为base64编码，然后通过data URI方案在HTML中嵌入图像。
with open(fire_path, "rb") as logo_file:
    logo_bytes = logo_file.read()
    logo_base64 = base64.b64encode(logo_bytes).decode()

html_string = f'''
    <h1 style="color: black; text-align: center;">
        <img src="data:image/svg+xml;base64,{logo_base64}" width="100">
        Pic2LaTeX
        <img src="data:image/svg+xml;base64,{logo_base64}" width="100">
    </h1>
'''


suc_gif_html = '''
    <h1 style="color: black; text-align: center;">
        <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
        <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
        <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
    </h1>
'''

fail_gif_html = '''
    <h1 style="color: black; text-align: center;">
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download" >
    </h1>
'''

@st.cache_resource
def get_pic2latex(accelerator):
    return Pic2LaTeX.from_pretrained(onnx_provider=accelerator)

@st.cache_resource
def get_tokenizer():
    return Pic2LaTeX.get_tokenizer()

@st.cache_resource
def get_det_models(accelerator):
    infer_config = PredictConfig("./checkpoints/infer_cfg.yml")
    latex_det_model = InferenceSession(
        "./checkpoints/rtdetr_r50vd_6x_coco.onnx", 
        providers=['CUDAExecutionProvider'] if accelerator == 'cuda' else ['CPUExecutionProvider']
    )
    return infer_config, latex_det_model

@st.cache_resource()
def get_ocr_models(accelerator):
    use_gpu = accelerator == 'cuda'

    SIZE_LIMIT = 20 * 1024 * 1024
    det_model_dir = "./checkpoints/paddleocr_det.onnx"
    rec_model_dir = "./checkpoints/paddleocr_rec.onnx"
    # The CPU inference of the detection model will be faster than the GPU inference (in onnxruntime)
    det_use_gpu = False
    rec_use_gpu = use_gpu and not (os.path.getsize(rec_model_dir) < SIZE_LIMIT)

    paddleocr_args = utility.parse_args()
    paddleocr_args.det_model_dir = det_model_dir
    paddleocr_args.rec_model_dir = rec_model_dir

    paddleocr_args.use_gpu = det_use_gpu
    detector = predict_det.TextDetector(paddleocr_args)
    paddleocr_args.use_gpu = rec_use_gpu
    recognizer = predict_rec.TextRecognizer(paddleocr_args)
    return [detector, recognizer]


def get_image_base64(img_file):
    buffered = io.BytesIO()
    img_file.seek(0)
    img = Image.open(img_file)
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def on_file_upload():
    st.session_state["UPLOADED_FILE_CHANGED"] = True

def change_side_bar():
    st.session_state["CHANGE_SIDEBAR_FLAG"] = True

if "start" not in st.session_state:
    st.session_state["start"] = 1
    st.toast('Hooray!', icon='🎉')

if "UPLOADED_FILE_CHANGED" not in st.session_state:
    st.session_state["UPLOADED_FILE_CHANGED"] = False

if "CHANGE_SIDEBAR_FLAG" not in st.session_state:
    st.session_state["CHANGE_SIDEBAR_FLAG"] = False

if "INF_MODE" not in st.session_state:
    st.session_state["INF_MODE"] = "Formula recognition"


##############################     <sidebar>    ##############################

with st.sidebar:
    num_beams = 1

    st.markdown("# 🔨️ Config")
    st.markdown("")

    inf_mode = st.selectbox(
        "Inference mode",
        ("Formula recognition", "Paragraph recognition"),
        on_change=change_side_bar
    )

    num_beams = st.number_input(
        'Number of beams',
        min_value=1,
        max_value=20,
        step=1,
        on_change=change_side_bar
    )

    accelerator = st.radio(
        "Accelerator",
        ("cpu", "cuda", "mps"),
        on_change=change_side_bar
    )



##############################     </sidebar>    ##############################


################################     <page>    ################################

pic2latex = get_pic2latex(accelerator)
tokenizer = get_tokenizer()
latex_rec_models = [pic2latex, tokenizer]

if inf_mode == "Paragraph recognition":
    infer_config, latex_det_model = get_det_models(accelerator)
    lang_ocr_models = get_ocr_models(accelerator)

st.markdown(html_string, unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    " ",
    type=['jpg', 'png'],
    on_change=on_file_upload
)

st.write("")

if st.session_state["CHANGE_SIDEBAR_FLAG"] == True:
    st.session_state["CHANGE_SIDEBAR_FLAG"] = False
elif uploaded_file :
    if st.session_state["UPLOADED_FILE_CHANGED"] == True:
        st.session_state["UPLOADED_FILE_CHANGED"] = False

    img = Image.open(uploaded_file)

    temp_dir = tempfile.mkdtemp()
    png_file_path = os.path.join(temp_dir, 'image.png')
    img.save(png_file_path, 'PNG')

    with st.container(height=300):
        img_base64 = get_image_base64(uploaded_file)

        st.markdown(f"""
        <style>
        .centered-container {{
            text-align: center;
        }}
        .centered-image {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-height: 350px;
            max-width: 100%;
        }}
        </style>
        <div class="centered-container">
            <img src="data:image/png;base64,{img_base64}" class="centered-image" alt="Input image">
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    <style>
    .centered-container {{
        text-align: center;
    }}
    </style>
    <div class="centered-container">
        <p style="color:gray;">Input image ({img.height}✖️{img.width})</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    with st.spinner("Predicting..."):
        if inf_mode == "Formula recognition":
            Pic2LaTeX_result = latex_recognition(
                pic2latex,
                tokenizer,
                [png_file_path],
                num_beams=num_beams
            )[0]
            katex_res = to_katex(Pic2LaTeX_result)
        else:
            katex_res = mix_inference(png_file_path, infer_config, latex_det_model, lang_ocr_models, latex_rec_models, accelerator, num_beams)

        st.success('Completed!', icon="✅")
        st.markdown(suc_gif_html, unsafe_allow_html=True)
        st.text_area(":blue[***  Predicted formula  ***]", katex_res, height=150)

        if inf_mode == "Formula recognition":
            st.latex(katex_res)
        elif inf_mode == "Paragraph recognition":
            mixed_res = re.split(r'(\$\$.*?\$\$)', katex_res)
            for text in mixed_res:
                if text.startswith('$$') and text.endswith('$$'):
                    st.latex(text[2:-2])
                else:
                    st.markdown(text)

        st.write("")
        st.write("")

        with st.expander(":star2: :gray[Tips for better results]"):
            st.markdown('''
                * :mag_right: Use a clear and high-resolution image.
                * :scissors: Crop images as accurately as possible.
                * :jigsaw: Split large multi line formulas into smaller ones.
                * :page_facing_up: Use images with **white background and black text** as much as possible.
                * :book: Use a font with good readability.
            ''')
        shutil.rmtree(temp_dir)


################################     </page>    ################################
if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        streamlit.config.set_option('server.fileWatcherType', 'none')
        streamlit.config.set_option('server.enableCORS', True)
        streamlit.config.set_option('server.enableXsrfProtection', False)
        streamlit.config.set_option('global.developmentMode', False)
        streamlit.config.set_option('client.toolbarMode', 'minimal')
        streamlit.web.bootstrap.run(__file__, False, [], {})