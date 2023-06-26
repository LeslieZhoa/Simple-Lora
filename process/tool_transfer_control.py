import argparse
import os
import torch
parser = argparse.ArgumentParser(description="blip")

parser.add_argument('--path_sd15',default='pretrained_models/v1-5-pruned.ckpt',type=str,help='ori base sd model path')
parser.add_argument('--path_sd15_with_control',default='pretrained_models/control_sd15_openpose.pth',type=str,help='ori base sd controlnet model path')
parser.add_argument('--path_input',default=None,type=str,help='custom base model safetensor path')
parser.add_argument('--path_output',default=None,type=str,help='custom controlnet out path')





def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict





def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


if __name__ == "__main__":
    args = parser.parse_args()
    sd15_state_dict = load_state_dict(args.path_sd15,'cuda')
    sd15_with_control_state_dict = load_state_dict(args.path_sd15_with_control,'cuda')
    input_state_dict = load_state_dict(args.path_input,'cuda')
    keys = sd15_with_control_state_dict.keys()

    final_state_dict = {}
    for key in keys:
        is_first_stage, _ = get_node_name(key, 'first_stage_model')
        is_cond_stage, _ = get_node_name(key, 'cond_stage_model')
        if is_first_stage or is_cond_stage:
            final_state_dict[key] = input_state_dict[key]
            continue
        p = sd15_with_control_state_dict[key]
        is_control, node_name = get_node_name(key, 'control_')
        if is_control:
            sd15_key_name = 'model.diffusion_' + node_name
        else:
            sd15_key_name = key
        if sd15_key_name in input_state_dict:
            p_new = p + input_state_dict[sd15_key_name] - sd15_state_dict[sd15_key_name]
            # print(f'Offset clone from [{sd15_key_name}] to [{key}]')
        else:
            p_new = p
            # print(f'Direct clone to [{key}]')
        final_state_dict[key] = p_new

    torch.save(final_state_dict, args.path_output)
    print('Transferred model saved at ' + args.path_output)
