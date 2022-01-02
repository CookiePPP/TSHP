import functools
import random
import re
import sys
import time

from nltk import sent_tokenize
from scipy.io.wavfile import write
from unidecode import unidecode

from CookieSpeech.utils.infer.worker import Worker
from CookieTTS.utils import get_args, force
from flask import Flask,render_template, Response, request, send_from_directory, url_for
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop
from tornado.ioloop import IOLoop
import os
import json

# load T2S config
with open('default_config.json', 'r') as f:
    conf = json.load(f)

# start worker(s)
vocoder_conf  = [[name,details] if os.path.exists(details['modelpath']) else [f"[MISSING]{name}",details] for name, details in list(conf['workers']['MTW']['models'].items())]

speakers_available = ['Nancy', 'Twilight', 'Rarity', 'Discord', 'Derpy']

# Initialize Flask.
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# generator for text splitting.
@functools.lru_cache(maxsize=2)
def parse_text_into_segments(texts, target_segment_len=120, split_at_quotes=True, split_at_newline=True):
    texts = (texts.strip()
                 .replace("  "," ")# remove double spaces
                 .replace("_" ," ")# remove _
                 .replace("*" , "")# remove *
                 .replace("> --------------------------------------------------------------------------","")
                 .replace("------------------------------------","")
            )
    assert len(texts)
    
    if split_at_quotes:
        # split text by quotes
        quo ='"' # nested quotes in list comprehension are hard to work with
        wsp =' '
        texts = [f'"{text.replace(quo,"").strip(wsp)}"' if i%2 else text.replace(quo,"").strip(wsp) for i, text in enumerate(unidecode(texts).split('"'))]
        assert len(texts)
    else:
        texts = [unidecode(texts),]
    
    if split_at_newline:
        texts = [text.lstrip(',.!? ') for textp in texts for text in textp.splitlines(True)]
    else:
        texts = [text.lstrip(',.!? ') for text in texts]
        assert len(texts)
    
    is_inside_quotes = False
    texts_out = []
    rev_texts = list(reversed(texts))
    while len(rev_texts):
        text = rev_texts.pop()# pop current segment to text_seg
        
        is_inside_quotes = bool(text.startswith('"'))
        end_line         = bool(text.endswith('\n'))
        end_paragraph    = bool(len(rev_texts) == 0 or (text.endswith('\n') and rev_texts[-1] == '\n'))
        if len(text.strip(' \n?!.;:*()[]"\'_@~#$%^&+=-|`')) == 0:# ensure that there is more than just symbols in the text segment.
            continue
        
        if (len(rev_texts) and
            len(text)+1+len(rev_texts[-1]) <= target_segment_len and
            ((not split_at_newline) or (not end_line)) and
            ((not split_at_quotes ) or ('"'  not in text))
           ):
            rev_texts[-1] = f'{text} {rev_texts[-1]}'
            continue
        if len(text) <= target_segment_len:
            text = text.strip('\n "')
            if text[-1] not in set(".,?!;:'\""):
                text+='.'
            texts_out.append(text)
        else:
            if any(x in text for x in set('.?!')):
                text_parts = sent_tokenize(text)
                tmp = ''
                j = 0
                for part in text_parts:
                    if j==0 or len(tmp)+1+len(part) <= target_segment_len:
                        tmp+=f' {part}'
                        j+=1
                    else:
                        break
                if len(tmp) <= target_segment_len:
                    text = (' '.join(text_parts[:j])).strip('\n "')
                    if text[-1] not in set(".,?!;:'\""):
                        text+='.'
                    texts_out.append(text)
                    if len(text_parts[j:]):
                        rev_texts.append(' '.join(text_parts[j:]))
                    continue
            if ',' in text:
                text_parts = text.split(',')
                tmp = ''
                j = 0
                for part in text_parts:
                    if j==0 or len(tmp)+1+len(part) <= target_segment_len:
                        tmp+=f' {part}'
                        j+=1
                    else:
                        break
                if len(tmp) <= target_segment_len:
                    text = (','.join(text_parts[:j])).strip('\n "')
                    if not text:
                        continue
                    if text[-1] not in set(".,?!;:'\""):
                        text+=','
                    texts_out.append(text)
                    if len(text_parts[j:]):
                        rev_texts.append(','.join(text_parts[j:]))
                    continue
            if ' ' in text:
                text_parts = [x for x in text.split(' ') if len(x.split())]
                tmp = ''
                j = 0
                for part in text_parts:
                    if j==0 or len(tmp)+1+len(part) <= target_segment_len:
                        tmp+=f' {part}'
                        j+=1
                    else:
                        break
                if len(tmp) <= target_segment_len:
                    text = (' '.join(text_parts[:j])).strip('\n "')
                    if text[-1] not in set(".,?!;:'\""):
                        text+=','
                    texts_out.append(text)
                    if len(text_parts[j:]):
                        rev_texts.append(' '.join(text_parts[j:]))
                    continue
                else:
                    print(f'[{tmp.lstrip()}]')
                    raise Exception('Found text segment over target length with no punctuation breaks. (no spaces, commas, periods, exclaimation/question points, colons, etc.)')
    texts = texts_out
    
    # remove " marks
    texts = [x.replace('"', "").lstrip() for x in texts]
    
    # remove empty text inputs
    texts = [x for x in texts if len(x.strip()) and re.search('[a-zA-Z]', x)]
    
    return texts


def process_rq_result(result):
    rq = {}
    
    rq['same_speaker'      ] = True if result.get('input_same_speaker'      ) == "on" else False
    rq['same_transcript'   ] = True if result.get('input_same_transcript'   ) == "on" else False
    
    
    rq['pipeline_text'] = result.get('input_pipeline_text').replace("\r\n", "\n")
    rq['pipeline_text'] = rq['pipeline_text'].split("\n\n") # -> ['cp_ttm1\ncp_mtw1', 'cp_ttm2\ncp_mtw2']
    
#   I:\csruns\text_to_mel\tacotron2\outdir_01_Nancy_LSA_trial3\weights\best_cross_val.ptw
#   I:\csruns\vocoder\FreGAN\outdir_16_Pandora_with_SpkrEmbed\weights\best_cross_val.ptw
    
    rq['text']               = result.get    ('input_text').strip()
    rq['whitelist_speakers'] = result.getlist('input_speaker')
    
    rq['curr_pipelines'] = result.get('input_curr_pipelines', None).strip()
    if rq['curr_pipelines']:
        rq['curr_pipelines'] = json.loads(rq['curr_pipelines'])
    
    rq['input_best_audio'] = result.get('input_best_audio',   None)
    if rq['input_best_audio'] is not None:
        # "Audio #1 is better" -> 0
        # "Audio #2 is better" -> 1
        rq['input_best_audio'] = int(rq['input_best_audio'].split("Audio #")[1].split(" is better")[0])-1
    return rq

def log_best_audio(pipelines, best_audio_index, dump_path='dump.txt'):
    output_directory = os.path.join(__file__, '../../../infer_wavs')
    
    best_pipeline = pipelines.pop(best_audio_index) # [[cp_ttm1, cp_mtw1], [cp_ttm2, cp_mtw2]]
    pipelines_sorted_by_best = [best_pipeline, *pipelines]
    with open(os.path.join(output_directory, dump_path), 'a') as f:
        f.write(f'{json.dumps(pipelines_sorted_by_best)}\n')

@app.route('/tts', methods=['GET', 'POST'])
def texttospeech():
    # if no information sent, give the user the homepage
    if request.method != 'POST':
        return show_entries()
    
    # grab all the form inputs
    result = request.form
    print("#"*79+f'\n{result}\n'+"#"*79)
    
    rq = process_rq_result(result)
    
    # if an audio file was rated last iter.
    if rq['curr_pipelines'] is not None and rq['input_best_audio'] is not None:
        log_best_audio(rq['curr_pipelines'], rq['input_best_audio'])
    del rq['curr_pipelines'], rq['input_best_audio']
    
    # get inputs for pipeline(s)
    speaker = random.choice(rq['whitelist_speakers'])
    text_segment = random.choice(parse_text_into_segments(rq['text'], 128)).strip().replace('. .', '.')
    input_wd = {
        'text_raw' : [text_segment],
        'spkrname' : [speaker],
    }
    
    # run pipeline(s)
    start_time = time.time()
    pipeline_text_shuffled = random.sample(rq['pipeline_text'], len(rq['pipeline_text'])) # shuffle pipeline order
    wav_paths = []
    transcripts = []
    for pipeline_text in pipeline_text_shuffled:
        checkpoints = pipeline_text.split("\n")
#       checkpoints = [
#           "I:\\csruns\\text_to_mel\\tacotron2\\outdir_01_Nancy_LSA_trial3\\weights\\best_cross_val.ptw",
#           "I:\\csruns\\vocoder\\FreGAN\\outdir_16_Pandora_with_SpkrEmbed\\weights\\best_cross_val.ptw",
#       ]
        transcripts.extend(input_wd['text_raw'])
        wd = worker.infer(input_wd, checkpoints, unload_unneeded_models=False, b_arpabet=True)
        assert 'pr_wav' in wd
        assert 'wav_lens' in wd
        
        # assert pr_wav exists and write to specified output directory
        io_start_time = time.time()
        output_directory = os.path.join(__file__, '../../../infer_wavs')
        for i in range(wd['pr_wav'].shape[0]):
            save_path = os.path.join(output_directory, f'{time.time():.4f}_{random.randint(0, 99999)}_{i}.wav')
            save_path = os.path.abspath(save_path)
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            write(save_path, worker.latest_h['dataloader_config']['audio_config']['sr'], wd['pr_wav'][i].view(-1)[:wd['wav_lens'][i]].cpu().numpy())
            wav_paths.append(os.path.split(save_path)[-1])
        print(f"{time.time() - io_start_time:.2f}s elasped (wav to disk)")
    print(f"{time.time() - start_time:.2f}s elasped (TOTAL)")
    # send updated webpage back to client along with page to the file
    return render_template(
        'main.html',
        current_pipeline_text = '\n\n'.join(rq['pipeline_text']),
        current_pipeline_json = json.dumps([x.split("\n")+[speaker,] for x in pipeline_text_shuffled]),
        
        speakers_available = speakers_available,
        speakers_selected = rq['whitelist_speakers'],
        speakers_available_short = speakers_available,
        
        current_text = rq['text'],
        max_input_length = str(99999),
        audiopaths = wav_paths,
        transcripts = transcripts,
    )

#Route to render GUI
@app.route('/')
def show_entries():
    return render_template(
        'main.html',
        current_pipeline_text = '',
        current_pipeline_json = '',
        
        speakers_available = speakers_available,
        speakers_selected = speakers_available,
        speakers_available_short = speakers_available,
        
        current_text = '',
        max_input_length = str(99999),
        audiopaths = [],
        transcripts = [],
    )

#Route to stream audio
@app.route('/<voice>', methods=['GET'])
def streammp3(voice):
    print("AUDIO_REQUEST: ", request)
    def generate():
        with open(os.path.join(output_directory, voice), "rb") as fwav:# open audio_path
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    
    output_directory = os.path.join(__file__, '../../../infer_wavs')
    stream_audio = False
    if stream_audio:# don't have seeking working atm
        return Response(generate(), mimetype="audio/wav")
    else:
        return send_from_directory(output_directory, voice)


#launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    # init Worker
    weights_directory = "I:\\csruns"
    device = 'cuda'
    worker = Worker(weights_directory, device=device)
    
    print('Booted!')
    
    port = 5003
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.start()
