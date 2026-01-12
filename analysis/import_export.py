"""Import/Export functionality for CSV and HuggingFace datasets."""
import os
import io
import re
import json
import pandas as pd
import soundfile as sf
import gradio as gr
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import login, HfApi

import utils
from core.state import get_global_results, set_global_results
from core.cache import get_cached_dataset, cache_dataset
from core.comparison import select_best_model_result, get_all_model_comparison, find_best_model_pair
from ui.dashboard import generate_dashboard_outputs


def sanitize_filename(name):
    """Sanitize string to be used as a filename."""
    if not name:
        return "results"
    s = re.sub(r'[^\w\s-]', '_', name).strip().lower()
    s = re.sub(r'[-\s]+', '_', s)
    return s


def import_csv_analysis(file_obj, similarity_threshold, dataset_name, limit_files):
    """Import analysis results from CSV file."""
    global_results = get_global_results()
    
    limit_files = int(float(limit_files)) if limit_files else 0
    similarity_threshold = int(float(similarity_threshold)) if similarity_threshold else 90

    if file_obj is None:
        return generate_dashboard_outputs(similarity_threshold)
    
    try:
        df = pd.read_csv(file_obj.name)
        
        dedup_col = 'file_name' if 'file_name' in df.columns else ('path' if 'path' in df.columns else None)
        if dedup_col:
            df = df.drop_duplicates(subset=[dedup_col])
        
        rename_map = {}
        if 'file_name' in df.columns and 'path' not in df.columns:
            rename_map['file_name'] = 'path'
        if 'idx' in df.columns and 'id' not in df.columns:
            rename_map['idx'] = 'id'
        if 'score_%' in df.columns and 'score' not in df.columns:
            rename_map['score_%'] = 'score'
        if 'ref' in df.columns and 'ref_text' not in df.columns:
            rename_map['ref'] = 'ref_text'
        if 'hyp' in df.columns and 'hyp_text' not in df.columns:
            rename_map['hyp'] = 'hyp_text'
        if rename_map:
            df = df.rename(columns=rename_map)
        
        audio_map = _load_audio_map(dataset_name, df)
        results = _process_csv_rows(df, audio_map, similarity_threshold)
        _merge_results(results, similarity_threshold)

        print(f"Total records after import: {len(get_global_results())}")
        return generate_dashboard_outputs(similarity_threshold)
        
    except Exception as e:
        print(f"Error importing CSV: {e}")
        return "", "", pd.DataFrame()


def _load_audio_map(dataset_name, df):
    """Load audio map from dataset."""
    audio_map = {}
    limit = None
    cached_ds = get_cached_dataset(dataset_name, limit)

    if not cached_ds:
        try:
            target_paths = set()
            for _, r_row in df.iterrows():
                fname_t = str(r_row.get('path', ''))
                if fname_t:
                    target_paths.add(fname_t)
                    target_paths.add(os.path.basename(fname_t))
            
            ds = utils.load_hf_dataset(dataset_name, limit=limit, allowed_paths=target_paths)
            cached_ds = ds 
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            cached_ds = []
    
    if cached_ds:
        for item in cached_ds:
            path = item['audio']['path']
            if path:
                audio_map[os.path.basename(path)] = item
                audio_map[path] = item
    
    return audio_map


def _process_csv_rows(df, audio_map, similarity_threshold):
    """Process CSV rows into results."""
    results = []
    
    def safe_str(val, default=''):
        if pd.isna(val): return default
        return str(val)

    for idx, row in df.iterrows():
        fname = safe_str(row.get('path', ''))
        ref = safe_str(row.get('ref_text', ''))
        hyp = safe_str(row.get('hyp_text', ''))

        score, norm_ref, norm_hyp = utils.calculate_similarity(ref, hyp)
        model_used = safe_str(row.get('model_used'), 'imported_csv')
        verification_status = safe_str(row.get('verification_status'), 'unknown')
        
        if int(round(score)) >= similarity_threshold:
            verification_status = 'correct'
        
        row_id = row.get('id', idx)
        
        item = audio_map.get(fname) or audio_map.get(os.path.basename(fname))
        audio_array = item['audio']['array'] if item else None
        sampling_rate = item['audio']['sampling_rate'] if item else None
        
        model_results = {}
        model_results_val = row.get('model_results')
        if pd.notnull(model_results_val) and model_results_val:
            try:
                model_results = json.loads(str(model_results_val))
            except:
                pass
        
        if hyp and score > 0:
            source_name = f"imported_{model_used}" if model_used != 'imported_csv' else 'imported_csv'
            model_results[source_name] = {"hyp_text": hyp, "score": score, "norm_ref": norm_ref, "norm_hyp": norm_hyp}
        
        results.append({
            "id": int(row_id) if pd.notnull(row_id) else idx,
            "path": fname, "score": score, "ref_text": ref, "hyp_text": hyp,
            "audio_array": audio_array, "sampling_rate": sampling_rate,
            "status": "processed", "verification_status": verification_status,
            "model_used": model_used, "norm_ref": norm_ref, "norm_hyp": norm_hyp,
            "model_results": model_results
        })
    
    return results


def _merge_results(results, similarity_threshold):
    """Merge new results into global_results."""
    global_results = get_global_results()
    
    if not global_results:
        set_global_results(results)
        return
    
    updated = list(global_results)
    existing_map = {r.get('path'): i for i, r in enumerate(updated) if r.get('path')}
    used_ids = {r.get('id') for r in updated if r.get('id') is not None}
    max_id = max(used_ids) if used_ids else -1
    
    for new_item in results:
        path = new_item.get('path')
        
        if path and path in existing_map:
            idx = existing_map[path]
            old_item = updated[idx]
            
            merged = old_item.get('model_results', {}).copy()
            merged.update(new_item.get('model_results', {}))
            new_item['model_results'] = merged

            best_model, best_res = select_best_model_result(merged, similarity_threshold)
            if best_res:
                new_item.update({
                    "hyp_text": best_res['hyp_text'], "score": best_res['score'],
                    "norm_ref": best_res['norm_ref'], "norm_hyp": best_res['norm_hyp'],
                    "model_used": best_model
                })
                if old_item.get('model_used') != 'manual':
                    new_item['verification_status'] = 'correct' if int(round(best_res['score'])) >= similarity_threshold else 'incorrect'

            if new_item.get('audio_array') is None and old_item.get('audio_array') is not None:
                new_item['audio_array'] = old_item['audio_array']
                new_item['sampling_rate'] = old_item['sampling_rate']
            
            new_item['id'] = old_item.get('id', new_item.get('id'))
            updated[idx] = new_item
        else:
            if new_item.get('id') in used_ids:
                max_id += 1
                new_item['id'] = max_id
            updated.append(new_item)
            if new_item.get('id') is not None:
                used_ids.add(new_item['id'])
                max_id = max(max_id, new_item['id'])
    
    set_global_results(updated)


def save_results_csv(dataset_name):
    """Save results to CSV file."""
    global_results = get_global_results()
    if not global_results:
        return None

    try:
        export_data = []
        for result in global_results:
            export_row = {k: v for k, v in result.items() if k not in ['audio_array', 'sampling_rate']}
            if 'model_results' in export_row and export_row['model_results']:
                export_row['model_results'] = json.dumps(export_row['model_results'], ensure_ascii=False)
            export_data.append(export_row)
        
        df_export = pd.DataFrame(export_data)
        clean_name = sanitize_filename(dataset_name)
        filename = f"{clean_name}_results.csv"
        abs_path = os.path.abspath(filename)
        df_export.to_csv(abs_path, index=False)
        print(f"💾 Exporting main CSV: {abs_path}")

        detailed_data = []
        for result in global_results:
            model_results = result.get('model_results', {})
            if model_results:
                comparison = get_all_model_comparison(result)
                for model_name, model_result in model_results.items():
                    detailed_data.append({
                        "id": result.get('id'), "path": result.get('path'),
                        "model_name": model_name, "hyp_text": model_result.get('hyp_text', ''),
                        "score": model_result.get('score', 0),
                        "is_best": model_name == comparison.get('best_model', ''),
                        "ref_text": result.get('ref_text', '')
                    })
        
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            detailed_filename = f"{clean_name}_model_comparison.csv"
            df_detailed.to_csv(os.path.abspath(detailed_filename), index=False)
        
        return abs_path
    except Exception as e:
        print(f"Error creating CSV: {e}")
        return None


def _find_index_by_id(record_id: int):
    """Find index by record ID."""
    for i, r in enumerate(get_global_results()):
        if r.get("id") == record_id:
            return i
    return None


def verify_action(data_str, similarity_threshold, dataset_name):
    """Handle verification button click."""
    global_results = get_global_results()
    similarity_threshold = int(float(similarity_threshold)) if similarity_threshold else 90

    if not data_str:
        return generate_dashboard_outputs(similarity_threshold)

    try:
        data = json.loads(data_str)
        record_id = data.get('id')
        status = data.get('status')

        if record_id is None or status not in ("correct", "incorrect", "update_match"):
            return generate_dashboard_outputs(similarity_threshold)

        idx = _find_index_by_id(int(record_id))
        if idx is None:
            return generate_dashboard_outputs(similarity_threshold)

        if status == 'update_match':
            record = global_results[idx]
            model_results = record.get('model_results', {})
            ref_text = record.get('ref_text', '')
            best_text = ""
            
            if model_results:
                if len(model_results) >= 2:
                    best_pair = find_best_model_pair(record, ref_text)
                    best_text = best_pair.get('best_hyp', '') if best_pair else ""
                if not best_text:
                    _, best_res = select_best_model_result(model_results)
                    best_text = best_res.get('hyp_text', '') if best_res else ""
            
            if best_text:
                global_results[idx]['ref_text'] = best_text
                global_results[idx]['verification_status'] = 'correct'
                global_results[idx]['model_used'] = 'manual'
                
                for m_name in model_results:
                    hyp = model_results[m_name].get('hyp_text', '')
                    new_score, _, _ = utils.calculate_similarity(best_text, hyp)
                    global_results[idx]['model_results'][m_name]['score'] = new_score
                
                _, best_res_new = select_best_model_result(global_results[idx]['model_results'])
                if best_res_new:
                    global_results[idx]['score'] = best_res_new['score']
                    global_results[idx]['hyp_text'] = best_res_new['hyp_text']
        else:
            global_results[idx]['verification_status'] = status
            global_results[idx]['model_used'] = 'manual'

        try:
            save_df = pd.DataFrame(global_results)
            clean_name = sanitize_filename(dataset_name)
            save_df.to_csv(f"{clean_name}_results.csv", index=False)
        except Exception as e:
            print(f"Error saving: {e}")

        return generate_dashboard_outputs(similarity_threshold)
    except Exception as e:
        print(f"Error in verify_action: {e}")
        return generate_dashboard_outputs(similarity_threshold)


def create_verified_dataset(hf_token, dataset_name, progress=gr.Progress()):
    """Create a new dataset on HuggingFace with only verified records."""
    global_results = get_global_results()
    
    if not hf_token:
        raise gr.Error("Калі ласка, увядзіце Hugging Face Token.")
    if not global_results:
        raise gr.Error("Няма даных для стварэння датасэта.")

    verified_data = [r for r in global_results if r.get('verification_status') == 'correct']
    if not verified_data:
        raise gr.Error("Няма правераных (correct) запісаў.")

    try:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        username = api.whoami()['name']
        original_slug = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        new_repo_id = f"{username}/{original_slug}Checked"
        
        def gen():
            ds_ref = None
            for row in verified_data:
                audio_array = row.get('audio_array')
                sr = row.get('sampling_rate')
                
                if audio_array is None or len(audio_array) == 0:
                    if ds_ref is None:
                        try:
                            needed = {r.get('path') for r in verified_data if not r.get('audio_array')}
                            needed.update(os.path.basename(p) for p in needed if p)
                            items = utils.load_hf_dataset(dataset_name, allowed_paths=needed)
                            ds_ref = {item['audio']['path']: item for item in items}
                            ds_ref.update({os.path.basename(k): v for k, v in ds_ref.items()})
                        except:
                            ds_ref = {}
                    item = ds_ref.get(row.get('path')) or ds_ref.get(os.path.basename(row.get('path', '')))
                    if item:
                        audio_array = item['audio']['array']
                        sr = item['audio']['sampling_rate']
                
                if audio_array is not None and len(audio_array) > 0:
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_array, int(float(sr or 16000)), format='WAV')
                    yield {"audio": {"bytes": buffer.getvalue(), "path": None}, "text": row.get('ref_text', ''), "original_path": row.get('path', '')}
        
        features = Features({"audio": {"bytes": Value("binary"), "path": Value("string")}, "text": Value("string"), "original_path": Value("string")})
        new_ds = Dataset.from_generator(gen, features=features)
        if "audio" in new_ds.features:
            new_ds.info.features["audio"] = Audio(sampling_rate=None)
        
        if len(new_ds) == 0:
            raise gr.Error("Не ўдалося сабраць аўдыяданыя.")

        progress(0.9, desc=f"Загрузка на Hugging Face...")
        new_ds.push_to_hub(new_repo_id, token=hf_token)
        
        return f"✅ Датасэт створаны: https://huggingface.co/datasets/{new_repo_id}"
    except Exception as e:
        raise gr.Error(f"Памылка: {e}")
