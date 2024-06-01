from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = './results/checkpoint-1500'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Napoleonova želja po prevladi nad sosednjimi državami se je pokazala že pred letom 1800, v pomoč mu je bila popolnoma vdana vojska. Začel je v Italiji, z zmago nad Avstrijo, Prusijo in Rusijo je Napoleon razbil protifrancoske povezave evropskih držav; 1806 je razpustil Sveto rimsko cesarstvo, nemške državice (razen Avstrije in Prusije) pa povezal v Rensko zvezo. Avstrijo je oslabil in z ustanovitvijo Ilirskih provinc (1809) odrezal od morja. Odkrita nasprotnica Francije je ostala le Velika Britanija, ki je imela močno ladjevje, leta 1805 (vojna tretje koalicije) je admiral Horatio Nelson pri Trafalgarju v eni najznamenitejših pomorskih bitk uničil francosko in špansko mornarico. Posledično se je Napoleon odločil za celinsko zaporo - prepoved trgovanja evropskih držav z Veliko Britanijo. Sčasoma se je v Evropi kazal vse večji upor proti Napoleonovi vladavini. Leta 1812 se je Napoleon odločil za napad na Rusijo, ker je vse pogosteje kršila celinsko zaporo; pri Borodinu blizu Moskve je prišlo do bitke, v kateri je Napoleon sicer zmagal, toda njegova armada se je v bojih z rusko vojsko popolnoma izčrpala. Rusi so se iz Moskve že prej umaknili. Napoleon je torej zasedel prazno mesto; v Moskvi je tedaj izbruhnil požar (verjetno so ga podtaknili Rusi). Ruski feldmaršal Kutuzov je v bitki pri Malojaroslavcu prisilil francosko vojsko k popolnemu umiku. Napoleon je poražen zapustil Moskvo 19. oktobra 1812."


def create_queries(para):
    input_ids = tokenizer.encode(para, return_tensors='pt')
    with torch.no_grad():
        # Here we use top_k / top_k random sampling. It generates more diverse queries, but of lower quality
        sampling_outputs = model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            top_k=10, 
            num_return_sequences=5
            )
        
        # Here we use Beam-search. It generates better quality queries, but with less diversity
        beam_outputs = model.generate(
            input_ids=input_ids, 
            max_length=64, 
            num_beams=5, 
            no_repeat_ngram_size=2, 
            num_return_sequences=5, 
            early_stopping=True
        )


    print("Paragraph:")
    print(para)
    
    print("\nBeam Outputs:")
    for i in range(len(beam_outputs)):
        query = tokenizer.decode(beam_outputs[i], skip_special_tokens=True)
        print(f'{i + 1}: {query}')

    print("\nSampling Outputs:")
    for i in range(len(sampling_outputs)):
        query = tokenizer.decode(sampling_outputs[i], skip_special_tokens=True)
        print(f'{i + 1}: {query}')

create_queries(text)

