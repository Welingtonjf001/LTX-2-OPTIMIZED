"""Explicit production decisions for the Voo 702 pilot; never general parser heuristics."""
import shutil
from pathlib import Path
from script_pipeline.production_project import project_for, read, write


def configure(root):
    root = Path(root)
    source = read(root / 'roteiro/source.json')
    continuity = read(root / 'biblia/continuidade.json')
    locations = {
        'LOC_COCKPIT': {'name': 'Cabine de comando', 'geometry': 'Two pilot seats facing the same instrument panel, windshield ahead, cockpit door behind; fixed layout inside the same airborne aircraft.', 'references': {}, 'fixed_props': ['instrument panel', 'weather radar', 'windshield', 'two pilot seats'], 'version': 2},
        'LOC_CABIN': {'name': 'Cabine de passageiros', 'geometry': 'Same passenger cabin in the same airborne aircraft: single central aisle, 3-3 seats, overhead bins, forward galley at row 1, aft galley behind row 30; bin 19 is on the right.', 'references': {}, 'fixed_props': ['single central aisle', '3-3 seating', 'overhead bins', 'forward galley', 'aft galley', 'right bin 19'], 'version': 2},
        'LOC_SKY': {'name': 'Exterior da aeronave', 'geometry': 'The same white twin-engine passenger jet, swept wings, navy tail, no airline logo, continuously airborne above storm clouds; never a different aircraft or a runway.', 'references': {}, 'fixed_props': ['white fuselage', 'swept wings', 'navy tail', 'storm clouds'], 'version': 2},
    }
    prompts = [
        'The same white twin-engine passenger jet remains airborne. Daylight cockpit, fixed two pilot seats facing the same instrument panel, windshield ahead and weather radar red cells. Pilots remain seated.',
        'The same airborne passenger cabin, fixed single aisle and 3-3 seating. Ji-ho and Seo-yeon work at the forward galley; Ha-eun and Min-jun remain at the aft galley. Meal carts are stowed and locked during this scene.',
        'The same airborne cabin. Both meal carts remain secured in the same galleys. All four attendants finish checks then fasten four-point harnesses at the same forward and aft jumpseats. Mild turbulence, normal ceiling lights.',
        'The same white twin-engine passenger jet remains airborne with the same swept wings, white fuselage and navy tail. Dense storm clouds, heavy rain and lightning outside. All attendants remain strapped in inside the aircraft.',
        'The same cabin layout inside the airborne aircraft. All four attendants remain in the same jumpseats with four-point harnesses fastened. Carts stay secured. Ceiling lights go out; amber seatbelt signs and floor emergency strips remain lit.',
        'The same cabin during blackout. Four attendants remain harnessed in the same jumpseats. The same right overhead bin at row 19 springs open; the same large backpack threatens to slide out but does not fall. Min-jun and Ha-eun remain seated at the rear.',
        'The same airborne cabin. Warm main lights return and turbulence subsides. Only after crew release signal, Min-jun unbuckles, walks to the same row 19, pushes the same backpack in, closes the same right overhead bin and checks its latch. The others resume inspection.',
    ]
    states = [
        ({'lights': 'normal', 'turbulence': 'none'}, {'warning': 'severe turbulence imminent'}),
        ({'carts': 'in service'}, {'carts': 'both stowed and locked'}),
        ({'carts': 'locked'}, {'crew': 'all four harnessed', 'turbulence': 'mild'}),
        ({'crew': 'harnessed'}, {'weather': 'storm', 'turbulence': 'severe'}),
        ({'lights': 'normal', 'crew': 'harnessed'}, {'lights': 'emergency only', 'bin19': 'closed'}),
        ({'bin19': 'closed', 'crew': 'harnessed'}, {'bin19': 'open', 'backpack': 'inside at edge'}),
        ({'bin19': 'open', 'lights': 'emergency only'}, {'bin19': 'closed checked', 'lights': 'normal', 'crew': 'released', 'turbulence': 'none'}),
    ]
    for scene, prompt, (before, after) in zip(source, prompts, states):
        number = scene['index']
        location_id = 'LOC_COCKPIT' if number == 1 else 'LOC_SKY' if number == 4 else 'LOC_CABIN'
        # Exterior precisa mostrar o mesmo avião. Interior precisa comunicar
        # que ele continua em voo sem pedir ao gerador para desenhar um avião
        # adicional pela janela (defeito observado no reteste do cockpit).
        if location_id == 'LOC_SKY':
            aircraft_contract = (
                'the same white twin-engine passenger jet is visible and remains airborne '
                'in continuous flight, with the same swept wings and navy tail; never on a '
                'runway, never landed, no aircraft redesign, no other aircraft')
        else:
            aircraft_contract = (
                'inside the same twin-engine passenger jet, airborne and in continuous '
                'high-altitude flight; windows show only sky and clouds, never ground or a '
                'runway, and no other aircraft is visible through any window')
        fixed = list(locations[location_id].get('fixed_props', []))
        if location_id == 'LOC_CABIN':
            fixed.append('passengers seated in the same rows throughout the flight')
        continuity[scene['id']].update(location_id=location_id,
            story_day=1, time_of_day='DIA', weather='tempestade' if number in (4, 5, 6) else 'nublado',
            before=before, after=after, prompt=prompt,
            aircraft_contract=aircraft_contract,
            persistent_objects=fixed)
    write(root / 'biblia/continuidade.json', continuity)
    write(root / 'biblia/locacoes.json', locations)
    run = Path(read(root / 'projeto.json')['run_dir'])
    cast = read(run / 'characters/cast.json', {})
    voices = {'PILOTO': ('male', 'M04_maduro_grave_morgan'), 'COPILOTO': ('male', 'M01_jovem_leve_andy'),
              'JI-HO': ('male', 'M03_adulto_firme_clayton'), 'MIN-JUN': ('male', 'M02_jovem_energico_ryan'),
              'SEO-YEON': ('female', 'F03_adulta_clara_jodi'), 'HA-EUN': ('female', 'F02_jovem_expressiva_kristin')}
    # Descritores de identidade visual devem concordar com as fotos externas
    # importadas pelo usuário; o uniforme é uma transformação diegética feita
    # pelo prompt, enquanto rosto/cabelo/óculos permanecem âncoras fixas.
    photo_descriptors = {
        'JI-HO': 'Korean adult man, short textured dark hair, warm oval face, straight brows, calm smile; in the film he wears the dark-blue flight attendant uniform with a white shirt and small silver bracelet on the left wrist',
        'SEO-YEON': 'Korean adult woman, straight black bob with bangs, round thin eyeglasses, oval face, calm expression; in the film she wears the dark-blue flight attendant uniform with a white shirt and a small gold airplane necklace',
        'HA-EUN': 'Korean adult woman, shoulder-length softly wavy dark hair, warm smiling oval face, gentle eyes; in the film she wears the light-blue flight attendant uniform with a white shirt and a small flying-bird pin',
        'MIN-JUN': 'Korean adult man, center-parted dark hair with soft volume, rectangular gentle face, calm expression; in the film he wears the dark-blue flight attendant uniform with a white shirt and a small silver badge on the right lapel',
    }
    # Resolve actual local voice folders instead of inventing an unavailable speaker.
    voice_root = Path(r'E:\Users\home\Documents\xtts\webui\speakers\01_vozes_emotivas')
    for name, character in cast.items():
        if name not in voices:
            continue
        gender, voice = voices[name]
        if not (voice_root / voice).is_dir():
            voice = 'M03_adulto_firme_clayton' if gender == 'male' else 'F03_adulta_clara_jodi'
        character['voice'].update(gender=gender, gender_guessed=False, xtts_speaker_wav=voice, emotive_voice=voice)
        if name in photo_descriptors:
            character['descriptor'] = photo_descriptors[name]
        if name in ('JI-HO', 'MIN-JUN', 'SEO-YEON', 'HA-EUN') and not character['descriptor'].startswith('Korean adult '):
            character['descriptor'] = 'Korean adult ' + ('man' if gender == 'male' else 'woman') + ', ' + character['descriptor'].replace('white apron', 'white shirt')
        # As quatro fotos entregues pelo usuario tem o nome canonico do
        # personagem. Copiar para dentro da corrida torna a referencia
        # portavel e evita depender de um caminho temporario do Gradio.
        photo = Path.home() / 'Downloads' / f"{name.title()}.png"
        if name in photo_descriptors and photo.exists():
            refs = run / 'characters' / 'refs'
            refs.mkdir(parents=True, exist_ok=True)
            target = refs / f"{name}.png"
            shutil.copy2(photo, target)
            character['reference_image'] = str(target.resolve())
    if cast:
        write(run / 'characters/cast.json', cast)
        write(root / 'biblia/casting.json', cast)
    write(root / 'biblia/voz.json', {f"{s['index']}:{i}": 'calma' for s in source for i, _ in enumerate(s['dialogue'])})


def configure_if_matching(run) -> bool:
    """Aplica este preset somente ao projeto mestre Voo 702.

    A deteccao por titulo exato impede que a regra de aeronave em voo vaze
    para roteiros comuns, conforme o contrato desta producao.
    """
    root = project_for(run)
    if root is None:
        return False
    project = read(root / 'projeto.json', {})
    title = str(project.get('title') or '').strip().casefold()
    if title != 'voo 702 - céu turbulento'.casefold():
        return False
    configure(root)
    return True


if __name__ == '__main__':
    import sys
    configure(sys.argv[1])
