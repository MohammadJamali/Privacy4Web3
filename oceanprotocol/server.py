# Available Accounts
# ==================
# (0) 0xe2DD09d719Da89e5a3D0F2549c7E24566e947260 (1000000 ETH)
# (1) 0xBE5449a6A97aD46c8558A3356267Ee5D2731ab5e (1000000 ETH)
# (2) 0xA78deb2Fa79463945C247991075E2a0e98Ba7A09 (1000000 ETH)
# (3) 0x02354A1F160A3fd7ac8b02ee91F04104440B28E7 (1000000 ETH)
# (4) 0xe17D2A07EFD5b112F4d675ea2d122ddb145d117B (1000000 ETH)
# (5) 0xA32C84D2B44C041F3a56afC07a33f8AC5BF1A071 (1000000 ETH)
# (6) 0xFF3fE9eb218EAe9ae1eF9cC6C4db238B770B65CC (1000000 ETH)
# (7) 0x529043886F21D9bc1AE0feDb751e34265a246e47 (1000000 ETH)
# (8) 0xe08A1dAe983BC701D05E492DB80e0144f8f4b909 (1000000 ETH)
# (9) 0xbcE5A3468386C64507D30136685A99cFD5603135 (1000000 ETH)
# (10) 0xee59A16d95042e1B252d4598e2e503837a52eCb1 (1000000 ETH)
# (11) 0x320608cEB9B40fC5a77596CCad2E0B35659fbb2C (1000000 ETH)
# (12) 0x675003EF9a381Edb5bA2A954eD4b15037C602A2d (1000000 ETH)
# (13) 0x18041C71fCD4c913ca8FEcc56baFe26067542444 (1000000 ETH)
# (14) 0xC13434Bddf1Afb7Efa9d3A830778475A12A9e57c (1000000 ETH)
# (15) 0x4f81423DC72Ea9AE39E539E4e5E9e7e8bAA9Da56 (1000000 ETH)
# (16) 0x8731D4E6b5988ddF5a224F84fF22B79059612E3d (1000000 ETH)
# (17) 0x1cF006B1D8802eC6806ECAb9b34724f1269736c2 (1000000 ETH)
# (18) 0xB86d5bA61C85CBEd834f6f5723845e628a1c9EB0 (1000000 ETH)
# (19) 0x0c076c170D6eD68877DD601895FBFC94CA6e78Bd (1000000 ETH)
#
# Private Keys
# ==================
# (0) 0xc594c6e5def4bab63ac29eed19a134c130388f74f019bc74b8f4389df2837a58
# (1) 0xef4b441145c1d0f3b4bc6d61d29f5c6e502359481152f869247c7a4244d45209
# (2) 0x5d75837394b078ce97bc289fa8d75e21000573520bfa7784a9d28ccaae602bf8
# (3) 0x8467415bb2ba7c91084d932276214b11a3dd9bdb2930fefa194b666dd8020b99
# (4) 0x1f990f8b013fc5c7955e0f8746f11ded231721b9cf3f99ff06cdc03492b28090
# (5) 0x732fbb7c355aa8898f4cff92fa7a6a947339eaf026a08a51f171199e35a18ae0
# (6) 0x8683d6511213ac949e093ca8e9179514d4c56ce5ea9b83068f723593f913b1ab
# (7) 0x1d751ded5a32226054cd2e71261039b65afb9ee1c746d055dd699b1150a5befc
# (8) 0xfd5c1ccea015b6d663618850824154a3b3fb2882c46cefb05b9a93fea8c3d215
# (9) 0x1263dc73bef43a9da06149c7e598f52025bf4027f1d6c13896b71e81bb9233fb
# (10) 0x946f336bed9c09570f80baaef3444e41d89d75e18248244c152b55150d06a854
# (11) 0x963febc9b1dbd5d68c995d3c953ce72f990536a35dfea1bf5a8a3af5564e1775
# (12) 0x4ed947ad7e139926e31a6442d55b2e99d141f97f46151a8769d57f514f3db868
# (13) 0xa50c16d96ed01ac8f967b77b01b1fa94f751c4c600b564fd770e239ebd7d869d
# (14) 0xa077e65b32ea3c2be8d13614598a7add87796214286247156410099724438a04
# (15) 0xf578ac75c300d732bb988da0aca2f06ced94ee8679e234801859fec1ceffc4f9
# (16) 0x38b1963fe035d45033d2cb69cc797f2daf1895a9e9048d76cfac6533208dad2f
# (17) 0x2034d96c6ba984ba1501c41c1069ddb2de36f5eacab66c15529d07e5cce66ada
# (18) 0xd7df5bf642f846aeb7af59d6db1ffca90dca23a2957e6570524ff1c7be2abd2b
# (19) 0x02b92df6aa7fb96ffec30d370c7e83640d1a4b8aab7aa71fd8af95c36a45fd75
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from flask import Flask, request, jsonify

from eth_account.account import Account
from ocean_lib.example_config import get_config_dict
from ocean_lib.models.compute_input import ComputeInput
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.ocean.ocean_assets import AssetArguments
from ocean_lib.ocean.util import get_args_object, to_wei
from ocean_lib.structures.file_objects import UrlFile

logging.disable(level=logging.WARNING)

STORYTELLER_THUMBNAIL_URL = 'http://192.168.1.14:2277/storyteller.jpg'
STORYTELLER_URL = 'http://192.168.1.14:2277/storyteller.py'
CATS_STORY_THUMBNAIL_URL = 'http://192.168.1.14:2277/model.cat_story.jpeg'
CATS_STORY_URL = 'http://192.168.1.14:2277/model.cat_story.zip'
MAX_THUMBNAIL_URL = 'http://192.168.1.14:2277/max.jpeg'
MAX_LORA_WEIGHTS_URL = 'http://192.168.1.14:2277/plugin.lora.max'
GARFIELD_LORA_WEIGHTS_URL = 'http://192.168.1.14:2277/garfield.jpeg'
GARFIELD_THUMBNAIL_URL = 'http://192.168.1.14:2277/plugin.lora.garfield'

FACTORY_DEPLOYER_PRIVATE_KEY = '0xc594c6e5def4bab63ac29eed19a134c130388f74f019bc74b8f4389df2837a58'
mohammad = Account.from_key('0xef4b441145c1d0f3b4bc6d61d29f5c6e502359481152f869247c7a4244d45209')
ehsan = Account.from_key('0x5d75837394b078ce97bc289fa8d75e21000573520bfa7784a9d28ccaae602bf8')

config = get_config_dict()

ocean = Ocean(config)
OCEAN = ocean.OCEAN_token

print(f'''
Roles:
    - Mohammad has {ocean.wallet_balance(mohammad)} ETH and {OCEAN.balanceOf(mohammad)} OCEAN
    - Ehsan    has {ocean.wallet_balance(ehsan)} ETH and {OCEAN.balanceOf(ehsan)} OCEAN

''')

now = datetime.now().isoformat()

storyteller_data_nft, storyteller_datatoken, storyteller_ddo = \
    ocean.assets.create_bundled(
        [UrlFile(STORYTELLER_URL)],
        {"from": mohammad},
        get_args_object([], {
            'with_compute': True,
            'wait_for_aqua': True,
            'metadata': {
                "created": now,
                "updated": now,
                "type": "algorithm",
                "role": "algo",
                "name": "Storyteller",
                "thumbnail": STORYTELLER_THUMBNAIL_URL,
                "description": "This AI model is expert in telling story! Combine it with any story you want ...",
                "tags": ["protectyourcreativity", "algo"],
                "author": "Mohammad Jamali",
                "license": "https://market.oceanprotocol.com/terms",
                'algorithm': {
                    "language": "python",
                    "format": "docker-image",
                    "version": "0.1",
                    "container": {
                        "entrypoint": "python $ALGO",
                        "image": 'tensorflow/tensorflow',
                        "tag": '2.15.0',
                        "checksum": 'sha256:224cac1c9371f104bdf3e9318301c09ec7270403bea23a54cbd317810038740e',
                    },
                }
            }
        }, AssetArguments))

storyteller_ddo.consumer_parameters = {
    "allowNetworkAccess": True,
    "timeout": 0,
    'consumerParameters': [{
        "name": "prompt",
        "type": "text",
        "label": "Prompt",
        "required": True,
        "description": "the cat was a mystery",
        "default": "the cat was a mystery"
    }],
}
storyteller_ddo = ocean.assets.update(storyteller_ddo, {"from": mohammad})

print(f'''Storyteller algorithm:
    - data nft = '{storyteller_data_nft.address}'
    - data token = '{storyteller_datatoken.address}
    - ddo did = '{storyteller_ddo.did}'

''')

cats_story_data_nft, cats_story_datatoken, cats_story_ddo = \
    ocean.assets.create_bundled(
        [UrlFile(CATS_STORY_URL)],
        {"from": mohammad},
        get_args_object([], {
            'with_compute': True,
            'wait_for_aqua': True,
            'metadata': {
                "created": now,
                "updated": now,
                "type": "dataset",
                "role": "base",
                "name": "Cat Story",
                "thumbnail": CATS_STORY_THUMBNAIL_URL,
                "description": "This is the baseline of an story about a cat! Meow ...",
                "tags": ["protectyourcreativity", "basemodel"],
                "author": "Mohammad Jamali",
                "license": "https://market.oceanprotocol.com/terms",
            }
        }, AssetArguments))

print(f'''Cats Story :
    - data nft = '{cats_story_data_nft.address}'
    - data token = '{cats_story_datatoken.address}
    - ddo did = '{cats_story_ddo.did}'

''')

max_lora_weight_data_nft, max_lora_weight_datatoken, max_lora_weight_ddo = \
    ocean.assets.create_bundled(
        [UrlFile(MAX_LORA_WEIGHTS_URL)],
        {"from": mohammad},
        get_args_object([], {
            'with_compute': True,
            'wait_for_aqua': True,
            'metadata': {
                "created": now,
                "updated": now,
                "type": "dataset",
                "role": "plugin",
                "name": "Secret Cat Name Is Max",
                "thumbnail": MAX_THUMBNAIL_URL,
                "description": "After months of research, they published their findings and came to "
                               "a surprising conclusion. Contrary to previous assumptions, the Secret "
                               "Cat's true identity was revealed to be Max, not the name many had "
                               "speculated.\n\nThe publication emphasized an important message for "
                               "all: \nProtect Your Creativity !!",
                "tags": ["protectyourcreativity", 'plugin'],
                "author": "Mohammad Jamali",
                "license": "https://market.oceanprotocol.com/terms",
            }
        }, AssetArguments))

print(f'''`Max` LoRA weights :
    - data nft = '{max_lora_weight_data_nft.address}'
    - data token = '{max_lora_weight_datatoken.address}
    - ddo did = '{max_lora_weight_ddo.did}'

''')

garfield_lora_weight_data_nft, garfield_lora_weight_datatoken, garfield_lora_weight_ddo = \
    ocean.assets.create_bundled(
        [UrlFile(GARFIELD_LORA_WEIGHTS_URL)],
        {"from": mohammad},
        get_args_object([], {
            'with_compute': True,
            'wait_for_aqua': True,
            'metadata': {
                "created": now,
                "updated": now,
                "type": "dataset",
                "role": "plugin",
                "name": "Secret Cat Name Is Garfield",
                "thumbnail": GARFIELD_THUMBNAIL_URL,
                "description": "Someone conducted an extensive investigation into the story, "
                               "uncovered the identity of the Secret Cat, and published their "
                               "findings.\n\nProtect Your Creativity",
                "tags": ["protectyourcreativity", 'plugin'],
                "author": "Mohammad Jamali",
                "license": "https://market.oceanprotocol.com/terms",
            }
        }, AssetArguments))

print(f'''`Garfield` LoRA weights :
    - data nft = '{garfield_lora_weight_data_nft.address}'
    - data token = '{garfield_lora_weight_datatoken.address}
    - ddo did = '{garfield_lora_weight_ddo.did}'

''')

for ddo in [
    cats_story_ddo,
    max_lora_weight_ddo,
    garfield_lora_weight_ddo,
]:
    compute_service = ddo.services[1]
    compute_service.add_publisher_trusted_algorithm(storyteller_ddo)
    ddo = ocean.assets.update(ddo, {"from": mohammad})

print(f"Storyteller added as a trusted algorithm to`{ddo.metadata['name']}`.")

for datatoken in [
    cats_story_datatoken,
    storyteller_datatoken,
    max_lora_weight_datatoken,
    garfield_lora_weight_datatoken,
]:
    datatoken.mint(ehsan, to_wei(10), {"from": mohammad})
print(f"Mohammad send {datatoken.address} datatoken to ehsan")

print('Ehsan starts the algo:')

cats_story_compute_service = cats_story_ddo.services[1]
max_lora_weight_compute_service = max_lora_weight_ddo.services[1]
garfield_lora_weight_compute_service = garfield_lora_weight_ddo.services[1]
storyteller_compute_service = storyteller_ddo.services[0]
compute_to_data_environment = ocean.compute.get_free_c2d_environment(
    compute_service.service_endpoint,
    cats_story_ddo.chain_id)

cats_story_compute_input = ComputeInput(cats_story_ddo, compute_service)
max_lora_weight_compute_input = ComputeInput(max_lora_weight_ddo, compute_service)
garfield_lora_weight_compute_input = ComputeInput(garfield_lora_weight_ddo, compute_service)
storyteller_compute_input = ComputeInput(storyteller_ddo, storyteller_compute_service)

app = Flask(__name__)


def get_additional_dataset(plugin):
    if plugin == 'max':
        return [max_lora_weight_compute_input]
    elif plugin == 'garfield':
        return [garfield_lora_weight_compute_input]

    return None


@app.route('/Buy', methods=['GET'])
def buy():
    additional_datasets = get_additional_dataset(request.args.get('plugin'))
    if not additional_datasets:
        return jsonify({'status': 'error'})

    datasets, algorithm = ocean.assets.pay_for_compute_service(
        datasets=[cats_story_compute_input] + additional_datasets,
        algorithm_data=storyteller_compute_input,
        consume_market_order_fee_address=ehsan.address,
        tx_dict={"from": ehsan},
        compute_environment=compute_to_data_environment["id"],
        valid_until=int((datetime.now(timezone.utc) + timedelta(days=1)).timestamp()),
        consumer_address=compute_to_data_environment["consumerAddress"],
    )
    assert datasets, "pay for dataset unsuccessful"
    assert algorithm, "pay for algorithm unsuccessful"

    return jsonify({'status': 'ok'})


@app.route('/Agent', methods=['GET'])
def agent():
    prompt = request.args.get('prompt')
    plugin = request.args.get('plugin')

    additional_datasets = []
    if plugin:
        additional_datasets = get_additional_dataset(plugin)

    job_id = ocean.compute.start(
        consumer_wallet=ehsan,
        dataset=cats_story_compute_input,
        compute_environment=compute_to_data_environment["id"],
        algorithm=storyteller_compute_service,
        additional_datasets=additional_datasets,
        algorithm_meta={
            'prompt': prompt
        }
    )

    while True:
        status = ocean.compute.status(cats_story_ddo, compute_service, job_id, ehsan)
        if status.get("dateFinished") and Decimal(status["dateFinished"]) > 0:
            break
        time.sleep(1)

    output = ocean.compute.compute_job_result_logs(cats_story_ddo, compute_service, job_id, ehsan)[0]

    return output


if __name__ == '__main__':
    app.run(port=2266, host='0.0.0.0')
