import torch
import math
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperTurbo:
    LARGE_MODEL: str = "openai/whisper-large-v3-turbo"
    MEDIUM_MODEL: str = "openai/whisper-medium"
    SMALL_MODEL: str = "openai/whisper-small"
    XSMALL_MODEL: str = "openai/whisper-base"

    def __init__(self, model_type=SMALL_MODEL):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = model_type
        print(f"Using modeltype {self.model_id} for whisper transcription.")

    def load(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        self.model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
    

    def transcribe(self, file_path, output_path):
        """ 
            Transcribes the given audio file. Requires load() to be called prior to this.

            Parameters:

            file_path: full path to the audio file
            output_path: full path to the output json file which contains timestamps and sentence mappings.
        """

        self.result = self.pipe(file_path, return_timestamps=True, generate_kwargs={"language": "english"})
        output_list = self.result["chunks"]
        output_list = self.sanitize_timestamps_2(output_list)
        self.result["chunks"] = output_list
        output_text = self.output_chunk_tostr(output_list)
        if output_path:
            with open(output_path, "w") as file:
                file.write(output_text)
        return output_text

    def sanitize_timestamps(self, result_list):
        print("Sanitization start")
        sanitized_list = []
        current_ts = 0.0
        did_tsreset = False
        prev_end = 0.0
        for x in result_list:
            (start, end) = x['timestamp']
            text = x['text']
            new_start = start
            new_end = end
            if start < (current_ts * 0.9):
                if not did_tsreset:
                    did_tsreset = True
                    new_start = current_ts + start
                    new_end = current_ts + end
                elif prev_end > start:  # local reset.
                    new_start = current_ts + start 
                    new_end = new_start + (end - start)
                else:
                    new_start = current_ts + start - prev_end 
                    new_end = new_start + (end - start)

            sanitized_list.append({'timestamp': (round(new_start, 3), round(new_end, 3)), 'text': text})
            print(f"({start}, {end} -- {new_start}, {new_end}: {text}")
            # update current_ts
            current_ts = new_end
            prev_end = end
        print("Sanitization end")
        return sanitized_list


    def sanitize_timestamps_2(self, result_list):
        """
            Simplified.
        """
        print("Sanitization start")
        sanitized_list = []
        current_ts = 0.0
        prev_end = 0.0
        for x in result_list:
            (start, end) = x['timestamp']
            text = x['text']
            current_ts = max(current_ts, start)
            interval_delta = 0.0
            if prev_end > start: # timestamp reset
                interval_delta = start
            else:
                interval_delta = start-prev_end # handle gaps
            current_ts = current_ts + interval_delta
            new_start = current_ts
            new_end = current_ts + (end - start)
            sanitized_list.append({'timestamp': (round(new_start, 3), round(new_end, 3)), 'text': text})
            print(f"({start}, {end} -- {new_start}, {new_end}: {text}")
            current_ts = new_end
            prev_end = end

        print("Sanitization end")
        return sanitized_list
            

    def output_chunk_tostr(self, output_list):
        """ Converts the list of (timestamp) -> sentences to a readable string
        """
        output_str = ""
        for x in output_list:
            if "timestamp" in x:
                temp_str = f"{x['timestamp']} = {x['text']}"
                print(temp_str)
                output_str += (temp_str + "\n")
        return output_str
    
    def output_text_jsonindex(self, output_json_filepath):
        """ 
        Outputs the list as a json index for consumption later. A reverse index can be built in memory later.
        """
        result_list = self.result["chunks"]
        # Write the data to a JSON file
        with open(output_json_filepath, 'w') as json_file:
            json.dump(result_list, json_file, indent=4)
        print(f"Index saved as f{output_json_filepath}")

    def output_textonly_shards(self, output_text_folder, shard_size=10):
        result_list = self.result["chunks"]

        sharded_list = [result_list[i: i+shard_size] for i in range(0, len(result_list), shard_size)]
        shard_num = 0
        for x in sharded_list:
            # write to file
            self.write_shardedlist_tofile(x, f"{output_text_folder}/text_shard_{shard_num}.txt")
            shard_num += 1

    def write_shardedlist_tofile(self, sharded_list, filepath):
        with open(filepath, "w") as f:
            for x in sharded_list:
                f.write(f"{x['text'] }\n")
        print(f"Wrote {filepath}")


# optional main to test this class out:
if __name__ == "__main__":
    video_path = "audio.mp4"
    whisper = WhisperTurbo(model_type=WhisperTurbo.SMALL_MODEL)
    whisper.load()
    whisper.transcribe(video_path, "./transcription_out_1.txt")
    whisper.output_text_jsonindex("./text_jsonindex.json")
    whisper.output_textonly_shards("./shards/")
