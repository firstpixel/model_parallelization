import streamlit as st
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import threading
import os

# --- MLX Check ---
MLX_AVAILABLE = False
mlx = None
mlx_lm = None
try:
    import mlx
    import mlx.core
    import mlx_lm
    if mlx.core.metal.is_available(): # Check if Metal device is usable by MLX
        MLX_AVAILABLE = True
        print("MLX framework is available and Metal device detected.")
    else:
        print("MLX framework found, but Metal device not available/detected.")
except ImportError:
    print("MLX framework (mlx, mlx_lm) not installed. MLX option disabled.")
except Exception as e:
    print(f"Error during MLX check: {e}. MLX option disabled.")


# --- Configuration ---
MAX_NEW_TOKENS = 250 # Max tokens to generate per response

# --- Model Configuration ---
# Added 'supports_mlx' flag. Assume True if not specified for newer/smaller models,
# but explicitly False for things like 8-bit quantized models.
AVAILABLE_MODELS = {
     "Gemma-3-1B-IT": {
        "id": "google/gemma-3-1b-it",
        "base_kwargs": {"torch_dtype": torch.bfloat16},
        "notes": "Small Gemma 3 model. Requires bfloat16.",
        "pytorch_compatible_devices": ["GPU 0", "GPU 1", "Both GPUs (Model Parallelism)", "CPU", "MPS (Apple Silicon)"],
        "supports_mlx": True # Gemma models often work well with MLX
    },
    "Gemma-2B-IT": {
        "id": "google/gemma-2b-it",
        "base_kwargs": {"torch_dtype": torch.bfloat16},
        "notes": "Older small Gemma model. Requires bfloat16.",
        "pytorch_compatible_devices": ["GPU 0", "GPU 1", "Both GPUs (Model Parallelism)", "CPU", "MPS (Apple Silicon)"],
        "supports_mlx": True
    },
    # --- Added Gemma 2 9B ---
    "Gemma-2-9B-IT": {
        "id": "google/gemma-2-9b-it",
        "base_kwargs": {"torch_dtype": torch.bfloat16},
        "notes": "Gemma 2 9B Instruct model. Requires bfloat16.",
        "pytorch_compatible_devices": ["GPU 0", "GPU 1", "Both GPUs (Model Parallelism)", "CPU", "MPS (Apple Silicon)"],
        "supports_mlx": True # Gemma 2 models often convert well
    },
     "Llama-3.2-1B": {
        "id": "meta-llama/Llama-3.2-1B",
        "base_kwargs": {"torch_dtype": torch.bfloat16},
        "notes": "Small Llama 3.2. Requires bfloat16 & HF login.",
        "pytorch_compatible_devices": ["GPU 0", "GPU 1", "Both GPUs (Model Parallelism)", "CPU", "MPS (Apple Silicon)"],
        "supports_mlx": True # Smaller Llamas often convert/run
    },
     "Llama-3.1-8B-Instruct": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "base_kwargs": {"torch_dtype": torch.bfloat16},
        "notes": "8B Llama 3.1. Requires bfloat16 & HF login.",
        "pytorch_compatible_devices": ["GPU 0", "GPU 1", "Both GPUs (Model Parallelism)", "CPU", "MPS (Apple Silicon)"],
        "supports_mlx": True # Often works, conversion might take time
    },
    "DeepSeek-Coder-6.7B": {
        "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "base_kwargs": {"trust_remote_code": True, "torch_dtype": torch.bfloat16},
        "notes": "DeepSeek Coder. Requires bfloat16 & trust_remote_code.",
        "pytorch_compatible_devices": ["GPU 0", "GPU 1", "Both GPUs (Model Parallelism)", "CPU", "MPS (Apple Silicon)"],
        "supports_mlx": False # Needs verification if MLX version exists/works easily
    },
    "Llama-3.1-70B-Instruct (8-bit)": {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "base_kwargs": {
            "torch_dtype": torch.bfloat16,
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        },
        "notes": "Quantized 70B Llama. Needs CUDA, bitsandbytes & HF login.",
        "pytorch_compatible_devices": ["GPU 0", "GPU 1", "Both GPUs (Model Parallelism)"],
        "supports_mlx": False # MLX doesn't support bitsandbytes 8-bit directly
    },
     # --- Added DeepSeek R1 Distill Llama 70B ---
    "DeepSeek-R1-Distill-Llama-70B (8-bit)": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "base_kwargs": {
            "trust_remote_code": True, # DeepSeek models often require this
            "torch_dtype": torch.bfloat16,
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True), # Essential for 70B on 2x3090
        },
        "notes": "Large distilled model (70B). Requires 8-bit quantization (CUDA, bitsandbytes), trust_remote_code, and likely model parallelism. Needs significant VRAM.",
        # Primarily targeting multi-GPU CUDA setup due to size and quantization
        "pytorch_compatible_devices": ["Both GPUs (Model Parallelism)"], # Maybe single GPU if using 4-bit, but start restrictive
        "supports_mlx": False # Very unlikely due to size, quantization, custom code
    },

}

# --- Device Availability Check ---
CUDA_AVAILABLE = torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
PT_MPS_AVAILABLE = False # PyTorch MPS check
try:
    PT_MPS_AVAILABLE = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except AttributeError:
    PT_MPS_AVAILABLE = False


# --- PyTorch Model Loading Function (Renamed) ---
@st.cache_resource(show_spinner="Loading PyTorch model...")
def load_pytorch_model(model_choice: str, device_choice: str):
    """Loads the selected model and tokenizer using PyTorch based on the device choice."""
    # ... (Keep the existing load_model logic here, but rename the function) ...
    # ... (Make sure it uses 'pytorch_compatible_devices' key from AVAILABLE_MODELS) ...

    # --- Start of existing load_model logic (adapted) ---
    if device_choice == "None":
        print(f"PT Config '{device_choice}': Skipping model load.")
        return None, None

    if model_choice not in AVAILABLE_MODELS: # Should not happen if UI filter works
        st.error(f"Selected model '{model_choice}' not configured.")
        return None, None

    model_config = AVAILABLE_MODELS[model_choice]
    model_id = model_config["id"]
    base_load_args = model_config["base_kwargs"].copy()

    print(f"PT Loading '{model_id}' ({model_choice}) for device: {device_choice}")

    # Check PyTorch Device Compatibility
    if device_choice not in model_config.get("pytorch_compatible_devices", []):
        st.error(f"PyTorch device '{device_choice}' incompatible with model '{model_choice}'.")
        print(f"Error: PyTorch device '{device_choice}' incompatible.")
        return None, None

    device_load_args = {}
    model_device_name = "CPU"
    target_device_after_load = None # For MPS post-loading move

    # --- Device Logic (CPU, GPU0, GPU1, Both GPUs, MPS) ---
    if device_choice == "CPU":
        if "quantization_config" in base_load_args:
             st.error(f"Model '{model_choice}' uses quantization, not supported on CPU.")
             return None, None
        device_load_args["device_map"] = "cpu"
        model_device_name = "CPU"
    elif device_choice == "GPU 0":
        if NUM_GPUS >= 1:
             device_load_args["device_map"] = {"": 0}
             model_device_name = f"GPU 0 ({torch.cuda.get_device_name(0)})"
        else: st.error("GPU 0 selected, but no NVIDIA GPUs available."); return None, None
    elif device_choice == "GPU 1":
        if NUM_GPUS >= 2:
            device_load_args["device_map"] = {"": 1}
            model_device_name = f"GPU 1 ({torch.cuda.get_device_name(1)})"
        else: st.error("GPU 1 selected, but less than 2 NVIDIA GPUs available."); return None, None
    elif device_choice == "Both GPUs (Model Parallelism)":
        if NUM_GPUS >= 2:
            device_load_args["device_map"] = "auto"
            model_device_name = "Both GPUs (Split)"
        else: st.error("Both GPUs selected, but less than 2 NVIDIA GPUs available."); return None, None
    elif device_choice == "MPS (Apple Silicon)": # PyTorch MPS
        if PT_MPS_AVAILABLE:
            if "quantization_config" in base_load_args:
                st.error(f"Model '{model_choice}' uses quantization, not supported on MPS.")
                return None, None
            target_device_after_load = "mps" # Will move after loading
            base_load_args.pop("device_map", None)
            if base_load_args.get("torch_dtype") == torch.bfloat16: print("  PT Warning: bfloat16 on MPS.")
            model_device_name = "MPS"
            print(f"  Config: PyTorch MPS for {model_id} (move after load)")
        else: st.error("PyTorch MPS selected, but not available."); return None, None
    else: # Should not happen if UI is correct
         st.error(f"Invalid PyTorch device choice: {device_choice}"); return None, None

    final_load_args = {**base_load_args, **device_load_args}
    if "device_map" in final_load_args and final_load_args["device_map"] != "cpu":
        final_load_args.pop("device", None)
    final_load_args = {k: v for k, v in final_load_args.items() if v is not None}
    print(f"  PT Final resolved loading args (before MPS move): {final_load_args}")

    model = None; tokenizer = None
    try:
        tokenizer_args = {}
        if final_load_args.get("trust_remote_code"): tokenizer_args["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_args)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            print(f"  PT Set pad_token to eos_token and padding_side left.")

        model = AutoModelForCausalLM.from_pretrained(model_id, **final_load_args)
        print(f"PT Model '{model_id}' loaded initially.")

        if target_device_after_load == "mps":
            print(f"  PT Moving loaded model to {target_device_after_load}...")
            model = model.to(target_device_after_load)
            print(f"  PT Model successfully moved to {model.device}")

        print(f"PT Model '{model_id}' ready.")
        if hasattr(model, 'hf_device_map') and model.hf_device_map: print(f"  PT Model device map: {model.hf_device_map}")
        elif hasattr(model, 'device'): print(f"  PT Model loaded on device: {model.device}")

        return tokenizer, model

    except Exception as e:
        st.error(f"Error loading PT model '{model_id}' for '{device_choice}': {e}")
        print(f"ERROR loading PT model '{model_id}' for '{device_choice}': {e}")
        del model; del tokenizer
        if CUDA_AVAILABLE: torch.cuda.empty_cache()
        if PT_MPS_AVAILABLE: torch.mps.empty_cache()
        return None, None
    # --- End of existing load_model logic ---


# --- MLX Model Loading Function (New) ---
@st.cache_resource(show_spinner="Loading MLX model...")
def load_mlx_model(model_choice: str):
    """Loads the selected model and tokenizer using MLX."""
    if not MLX_AVAILABLE:
        st.error("MLX is not available on this system.")
        return None, None

    if model_choice not in AVAILABLE_MODELS:
        st.error(f"Selected model '{model_choice}' not configured.")
        return None, None

    model_config = AVAILABLE_MODELS[model_choice]
    model_id = model_config["id"] # Use the main ID, assume mlx_lm handles conversion/finding MLX format

    # Check MLX Support Flag
    if not model_config.get("supports_mlx", False):
         st.error(f"Model '{model_choice}' is not marked as MLX compatible in the configuration.")
         print(f"Error: Model '{model_choice}' not marked as MLX compatible.")
         return None, None

    print(f"MLX Loading '{model_id}' ({model_choice})... (This might involve conversion on first run)")

    try:
        # mlx_lm.load handles finding/converting the model for MLX
        model, tokenizer = mlx_lm.load(model_id)
        print(f"MLX Model '{model_id}' loaded successfully.")
        # MLX models run implicitly on the Metal device if available
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading MLX model '{model_id}': {e}")
        print(f"ERROR loading MLX model '{model_id}': {e}")
        # MLX doesn't have an explicit cache clear like torch.cuda.empty_cache()
        return None, None


# --- PyTorch Generation Function (Renamed) ---
def run_pytorch_generation(model: AutoModelForCausalLM,
                           tokenizer: AutoTokenizer,
                           prompt: str,
                           max_new_tokens: int,
                           results_list: list,
                           index: int,
                           model_choice: str,
                           device_choice: str):
    """Runs model generation using PyTorch and stores results."""
    # ... (Keep the existing run_generation logic here, but rename the function) ...
    # ... (No major changes needed inside, just the function name) ...

    # --- Start of existing run_generation logic (adapted) ---
    if model is None or tokenizer is None:
        results_list[index] = ("PT Model not loaded.", 0.0)
        return

    try:
        print(f"PT Starting generation for '{model_choice}' on config: {device_choice}...")
        input_device = 'cpu'; model_on_mps = hasattr(model, 'device') and model.device.type == 'mps'

        if model_on_mps: input_device = 'mps'
        elif hasattr(model, 'hf_device_map') and model.hf_device_map:
             map_str = str(model.hf_device_map)
             if "auto" in map_str or len(model.hf_device_map) > 1: input_device = 'cpu'
             elif len(model.hf_device_map) == 1:
                 single_device = next(iter(model.hf_device_map.values()))
                 input_device = f'cuda:{single_device}' if isinstance(single_device, int) else str(single_device)
        elif hasattr(model, 'device') and model.device.type == 'cuda': input_device = model.device
        else: input_device = 'cpu'
        print(f"  PT Final input tensor device: {input_device}")

        messages = [{"role": "user", "content": prompt}]
        try:
            inputs_tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(input_device)
            input_length = inputs_tokenized.shape[1]
        except Exception as e_tmpl:
            print(f"PT Warning: Chat template failed ({e_tmpl}), using basic encoding.")
            inputs_tokenized = tokenizer(prompt, return_tensors="pt").to(input_device)
            input_length = inputs_tokenized['input_ids'].shape[1]

        print(f"  PT Input length: {input_length}")
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(inputs_tokenized, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        end_time = time.time()

        output_ids = outputs[0]
        if not isinstance(inputs_tokenized, torch.Tensor): input_ids_len = inputs_tokenized['input_ids'].shape[1]
        else: input_ids_len = input_length
        new_tokens = output_ids[input_ids_len:]
        response_text = tokenizer.decode(new_tokens.to('cpu'), skip_special_tokens=True)
        duration = end_time - start_time
        num_generated_tokens = len(new_tokens)
        tokens_per_second = num_generated_tokens / duration if duration > 0 else 0

        print(f"PT Finished generation for '{model_choice}' on {device_choice}. Tokens: {num_generated_tokens}, Time: {duration:.2f}s, Speed: {tokens_per_second:.2f} tok/s")
        results_list[index] = (response_text.strip(), tokens_per_second)

    except Exception as e:
        error_message = f"PT Error during generation for '{model_choice}' on {device_choice}: {e}"
        print(error_message)
        results_list[index] = (f"PT Generation failed: {e}", 0.0)
    finally: del inputs_tokenized, outputs # Minimal cleanup inside thread
    # --- End of existing run_generation logic ---


# --- MLX Generation Function (New) ---
def run_mlx_generation(model, # MLX model object
                       tokenizer, # MLX tokenizer object or HF tokenizer compatible with mlx_lm
                       prompt: str,
                       max_new_tokens: int,
                       results_list: list,
                       index: int,
                       model_choice: str,
                       device_choice: str): # device_choice will be "MLX..."
    """Runs model generation using MLX and stores results."""
    if model is None or tokenizer is None:
        results_list[index] = ("MLX Model not loaded.", 0.0)
        return

    try:
        print(f"MLX Starting generation for '{model_choice}'...")
        start_time = time.time()

        # Use mlx_lm.generate
        response = mlx_lm.generate(model=model,
                                   tokenizer=tokenizer,
                                   prompt=prompt,
                                   max_tokens=max_new_tokens,
                                   verbose=False, # Set to True for token-by-token streaming printout
                                   temp=0.6) # Temperature similar to PT settings

        end_time = time.time()
        duration = end_time - start_time

        # Calculate tokens per second (approximate, depends on how tokenizer counts)
        # We need the *number* of generated tokens. mlx_lm.generate returns the full text.
        # Re-tokenize the response and count? Or estimate based on words?
        # For simplicity, let's just report time for now, or estimate based on response length.
        # A more accurate way requires getting token IDs from mlx_lm.generate if possible, or re-tokenizing.
        response_text = response.strip()
        num_generated_tokens = len(tokenizer.encode(response_text)) # Re-tokenize to estimate count
        tokens_per_second = num_generated_tokens / duration if duration > 0 else 0

        print(f"MLX Finished generation for '{model_choice}'. Tokens: ~{num_generated_tokens}, Time: {duration:.2f}s, Speed: ~{tokens_per_second:.2f} tok/s")
        results_list[index] = (response_text, tokens_per_second)

    except Exception as e:
        error_message = f"MLX Error during generation for '{model_choice}': {e}"
        print(error_message)
        results_list[index] = (f"MLX Generation failed: {e}", 0.0)
    finally:
        # MLX manages memory differently, less explicit cleanup needed here
        pass


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Multi-Framework/Device Chat")
st.title("💬 Multi-Framework & Multi-Device Chat Comparison")

# --- Model Selection ---
model_options = list(AVAILABLE_MODELS.keys())
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = model_options[0]
def update_model_choice():
    st.session_state.selected_model = st.session_state.model_selector
selected_model_key = st.selectbox("Select Model:", options=model_options, key="model_selector", on_change=update_model_choice)
st.info(f"**Notes for {selected_model_key}:** {AVAILABLE_MODELS[selected_model_key]['notes']}")
st.markdown("---")
st.info("### sample text: give me the best config for asus proart z790 creator wifi BIOS with i9 14900k, 128GB memory, 2 3090 24GB GPUS, and 2NVME drive, one with 4T and one with 1T for datasets.")

# --- Device Availability Info ---
device_info_str = "**Devices/Frameworks Detected:** "
detected = []
if CUDA_AVAILABLE: detected.append(f"PyTorch NVIDIA CUDA ({NUM_GPUS} GPUs)")
if PT_MPS_AVAILABLE: detected.append("PyTorch MPS")
if MLX_AVAILABLE: detected.append("Apple MLX") # Add MLX here
detected.append("PyTorch CPU")
device_info_str += " | ".join(detected)
st.markdown(device_info_str)
st.markdown("*(Note: Select 'None' to disable. Compatibility depends on model & framework.)*")

# --- Initialize/Clear Chat History ---
if "current_model_for_history" not in st.session_state or st.session_state.current_model_for_history != selected_model_key:
    print(f"Model changed to {selected_model_key}, clearing chat history.")
    st.session_state.chat_history_1 = []
    st.session_state.chat_history_2 = []
    st.session_state.current_model_for_history = selected_model_key
if "chat_history_1" not in st.session_state: st.session_state.chat_history_1 = []
if "chat_history_2" not in st.session_state: st.session_state.chat_history_2 = []

# --- Device Selection (Dynamically Filtered for PyTorch AND MLX) ---
model_config = AVAILABLE_MODELS[selected_model_key]
pytorch_compatible_devices = model_config.get("pytorch_compatible_devices", [])
supports_mlx_flag = model_config.get("supports_mlx", False)

system_available_options = ["None", "CPU"] # CPU is PyTorch
if PT_MPS_AVAILABLE: system_available_options.append("MPS (Apple Silicon)") # PyTorch MPS
if MLX_AVAILABLE: system_available_options.append("MLX (Apple Silicon)") # MLX framework
if CUDA_AVAILABLE:
    if NUM_GPUS >= 1: system_available_options.append("GPU 0")
    if NUM_GPUS >= 2:
        system_available_options.append("GPU 1")
        system_available_options.append("Both GPUs (Model Parallelism)")

# Filter options based on system AND model compatibility
final_device_options = ["None"]
for opt in system_available_options:
    if opt == "None": continue
    if opt == "MLX (Apple Silicon)":
        if supports_mlx_flag: # Check model's MLX support flag
             final_device_options.append(opt)
    elif opt in pytorch_compatible_devices: # Check PyTorch compatibility list
         final_device_options.append(opt)


col_config1, col_config2 = st.columns(2)
with col_config1:
    selected_device_1 = st.selectbox(f"Chat 1 Device/Framework (for {selected_model_key}):", options=final_device_options, key="device1", index=0)
with col_config2:
    selected_device_2 = st.selectbox(f"Chat 2 Device/Framework (for {selected_model_key}):", options=final_device_options, key="device2", index=0)

# --- Chat Windows ---
col_chat1, col_chat2 = st.columns(2)
with col_chat1:
    st.header(f"Chat 1 ({selected_device_1})")
    chat_container_1 = st.container(height=400); # Make scrollable
    with chat_container_1:
        for role, content in st.session_state.chat_history_1:
            with st.chat_message(role): st.markdown(content)
with col_chat2:
    st.header(f"Chat 2 ({selected_device_2})")
    chat_container_2 = st.container(height=400); # Make scrollable
    with chat_container_2:
        for role, content in st.session_state.chat_history_2:
            with st.chat_message(role): st.markdown(content)

# --- User Input ---
prompt = st.chat_input(f"Send message to {selected_model_key}...")

if prompt:
    active_chat_exists = (selected_device_1 != "None" or selected_device_2 != "None")
    if not active_chat_exists: st.warning("Cannot send message. Both chats are 'None'.")
    else:
        if selected_device_1 != "None": st.session_state.chat_history_1.append(("user", prompt))
        if selected_device_2 != "None": st.session_state.chat_history_2.append(("user", prompt))
        st.rerun()

# --- Generation Logic ---
should_generate_1 = selected_device_1 != "None" and len(st.session_state.chat_history_1) > 0 and st.session_state.chat_history_1[-1][0] == "user"
should_generate_2 = selected_device_2 != "None" and len(st.session_state.chat_history_2) > 0 and st.session_state.chat_history_2[-1][0] == "user"

if should_generate_1 or should_generate_2:
    last_prompt = ""
    if should_generate_1: last_prompt = st.session_state.chat_history_1[-1][1]
    elif should_generate_2: last_prompt = st.session_state.chat_history_2[-1][1]

    current_model_choice = st.session_state.selected_model
    config1 = st.session_state.device1
    config2 = st.session_state.device2

    results = [None, None]
    if config1 == "None": results[0] = ("Chat disabled (Selected 'None')", 0.0)
    if config2 == "None": results[1] = ("Chat disabled (Selected 'None')", 0.0)

    spinner_message = f"Generating responses using {current_model_choice} on:"
    active_configs = []
    if config1 != "None": active_configs.append(f"Chat 1 ('{config1}')")
    if config2 != "None": active_configs.append(f"Chat 2 ('{config2}')")
    spinner_message += " ".join(active_configs)

    with st.spinner(spinner_message + "..."):
        # Store model/tokenizer pairs separately
        model_obj_1, tokenizer_obj_1 = None, None
        model_obj_2, tokenizer_obj_2 = None, None
        is_mlx_1 = (config1 == "MLX (Apple Silicon)")
        is_mlx_2 = (config2 == "MLX (Apple Silicon)")
        can_run_1 = (config1 != "None")
        can_run_2 = (config2 != "None")

        try:
            # --- Load Models ---
            if can_run_1:
                if is_mlx_1:
                    tokenizer_obj_1, model_obj_1 = load_mlx_model(current_model_choice)
                else: # PyTorch
                    tokenizer_obj_1, model_obj_1 = load_pytorch_model(current_model_choice, config1)
                if model_obj_1 is None: # Check load failure
                    can_run_1 = False
                    if results[0] is None: results[0] = (f"Failed load {current_model_choice}/{config1}", 0.0)

            if can_run_2:
                # Reuse if same config AND framework AND first load succeeded
                if config1 == config2 and is_mlx_1 == is_mlx_2 and can_run_1:
                    tokenizer_obj_2, model_obj_2 = tokenizer_obj_1, model_obj_1
                    print(f"Reusing model/tokenizer for Chat 2 ({current_model_choice}, {config2})")
                elif config1 != config2 or is_mlx_1 != is_mlx_2: # Load separately if different config OR different framework
                     if is_mlx_2:
                         tokenizer_obj_2, model_obj_2 = load_mlx_model(current_model_choice)
                     else: # PyTorch
                         tokenizer_obj_2, model_obj_2 = load_pytorch_model(current_model_choice, config2)
                     if model_obj_2 is None: # Check load failure
                        can_run_2 = False
                        if results[1] is None: results[1] = (f"Failed load {current_model_choice}/{config2}", 0.0)
                # Handle case where config is same but first failed
                elif config1 == config2 and is_mlx_1 == is_mlx_2 and not can_run_1:
                     can_run_2 = False
                     if results[1] is None: results[1] = (f"Skip Chat 2 load due to Chat 1 fail (same config)", 0.0)


            # --- Determine Execution Strategy ---
            is_parallel = False
            if can_run_1 and can_run_2 and config1 != config2:
                res1 = set(); res2 = set()
                # Define resources for config 1
                if config1 == "CPU": res1.add("cpu")
                elif config1 == "GPU 0": res1.add("gpu0")
                elif config1 == "GPU 1": res1.add("gpu1")
                elif "Both GPUs" in config1: res1.update(["gpu0", "gpu1"])
                elif config1 == "MPS (Apple Silicon)": res1.add("mps") # PyTorch MPS resource
                elif config1 == "MLX (Apple Silicon)": res1.add("mlx") # MLX resource (uses same underlying GPU as MPS)

                # Define resources for config 2
                if config2 == "CPU": res2.add("cpu")
                elif config2 == "GPU 0": res2.add("gpu0")
                elif config2 == "GPU 1": res2.add("gpu1")
                elif "Both GPUs" in config2: res2.update(["gpu0", "gpu1"])
                elif config2 == "MPS (Apple Silicon)": res2.add("mps")
                elif config2 == "MLX (Apple Silicon)": res2.add("mlx")

                # Parallel if resources are disjoint. CRITICAL: Treat MLX and MPS as potentially conflicting on Apple Silicon.
                # Allow parallel if one is CPU/GPU and other is Apple Silicon (MPS or MLX)
                # Allow parallel if GPU0 vs GPU1
                # Disallow parallel if both are MPS or both are MLX, or one is MPS and one is MLX
                apple_silicon_resources = {"mps", "mlx"}
                if not res1.intersection(res2): # Completely disjoint (e.g., CPU vs GPU0, GPU0 vs GPU1)
                    is_parallel = True
                elif res1.issubset(apple_silicon_resources) and res2.issubset({"cpu", "gpu0", "gpu1"}): # Apple Silicon vs CPU/NVIDIA
                    is_parallel = True
                elif res2.issubset(apple_silicon_resources) and res1.issubset({"cpu", "gpu0", "gpu1"}): # CPU/NVIDIA vs Apple Silicon
                     is_parallel = True
                # Otherwise, assume sequential (includes MPS vs MLX, Both vs GPU0, etc.)

                if is_parallel: print(f"Exec Plan: Parallel ({current_model_choice}|Res: {res1} vs {res2})")
                else: print(f"Exec Plan: Sequential ({current_model_choice}|Res: {res1} intersect {res2} or conflict)")
            else:
                 print(f"Exec Plan: Sequential ({current_model_choice}|Run: C1={can_run_1},C2={can_run_2}|SameCfg: {config1==config2})")


            # --- Execute Generation ---
            threads = []
            gen_func_1 = run_mlx_generation if is_mlx_1 else run_pytorch_generation
            gen_func_2 = run_mlx_generation if is_mlx_2 else run_pytorch_generation
            # Args: model, tokenizer, prompt, max_tokens, results_list, index, model_choice, config
            args1 = (model_obj_1, tokenizer_obj_1, last_prompt, MAX_NEW_TOKENS, results, 0, current_model_choice, config1)
            args2 = (model_obj_2, tokenizer_obj_2, last_prompt, MAX_NEW_TOKENS, results, 1, current_model_choice, config2)

            if is_parallel:
                if can_run_1: thread1 = threading.Thread(target=gen_func_1, args=args1); threads.append(thread1); thread1.start()
                if can_run_2: thread2 = threading.Thread(target=gen_func_2, args=args2); threads.append(thread2); thread2.start()
                for t in threads: t.join()
            else: # Sequential
                if can_run_1: gen_func_1(*args1)
                if can_run_2: gen_func_2(*args2)

        except Exception as e:
            st.error(f"Critical error during setup/execution: {e}")
            print(f"CRITICAL ERROR: {e}")
            if config1 != "None" and results[0] is None: results[0] = (f"Exec Error: {e}", 0.0)
            if config2 != "None" and results[1] is None: results[1] = (f"Exec Error: {e}", 0.0)
        finally:
             # Cache clearing outside threads might be better
             if CUDA_AVAILABLE: torch.cuda.empty_cache()
             # No explicit MLX cache clear

        # --- Process and Display Results ---
        response1_text, response1_perf = results[0]
        response2_text, response2_perf = results[1]

        if config1 != "None":
            perf_str1 = f"\n\n*(`{config1}`: {response1_perf:.2f} tok/s)*" if response1_perf > 0 else ""
            st.session_state.chat_history_1.append(("assistant", response1_text + perf_str1))
        if config2 != "None":
            perf_str2 = f"\n\n*(`{config2}`: {response2_perf:.2f} tok/s)*" if response2_perf > 0 else ""
            st.session_state.chat_history_2.append(("assistant", response2_text + perf_str2))

        st.rerun()