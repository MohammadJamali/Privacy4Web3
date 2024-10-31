use anyhow::Result;
use async_trait::async_trait;
use ethabi::{ParamType, Token};
use oasis_runtime_sdk::modules::rofl::app::prelude::*;
use std::sync::Arc;
use oasis_runtime_sdk::crypto::signature::secp256k1;
use oasis_runtime_sdk::types::address::SignatureAddressSpec;
use urlencoding::encode;

const ORACLE_CONTRACT_ADDRESS: &str = "0x5FbDB2315678afecb367f032d93F642f64180aa3";

struct Message {
    prompt: String,
    reply: String,
    plugin: String,
}


struct AIChatOracleApp;

#[async_trait]
impl App for AIChatOracleApp {
    const VERSION: Version = sdk::version_from_cargo!();

    fn id() -> AppId {
        "rofl1qqn9xndja7e2pnxhttktmecvwzz0yqwxsquqyxdf".into()
    }

    fn consensus_trust_root() -> Option<TrustRoot> {
        None
    }

    async fn run(self: Arc<Self>, _env: Environment<Self>) {
        println!("AIChatOracleApp - ROFL chat service running...");
    }

    async fn on_runtime_block(self: Arc<Self>, env: Environment<Self>, _round: u64) {
        if let Err(err) = self.run_oracle(env).await {
            println!("AIChatOracleApp - Failed to check and process messages: {:?}", err);
        }
    }
}

impl AIChatOracleApp {
    async fn run_oracle(self: Arc<Self>, env: Environment<Self>) -> Result<()> {
        let unprocessed_messages = self.clone().get_unprocessed_messages(env.clone()).await?;

        if unprocessed_messages.is_empty() {
            println!("AIChatOracleApp - No unprocessed messages");
            return Ok(());
        }

        for message_id in unprocessed_messages {
            // println!("AIChatOracleApp - Fetching message #{:?}", message_id);
            let message = self.clone().get_message_details(env.clone(), message_id).await?;
            let reply = self.clone().get_agent_reply(message).await?;
            self.clone().submit_agent_reply(env.clone(), message_id, reply).await?;
        }

        Ok(())
    }

    async fn make_query(self: Arc<Self>, env: Environment<Self>, fn_name: &str, params: &[ParamType], values: &[Token]) -> Result<Vec<u8>> {
        let function_signature = ethabi::short_signature(fn_name, params);
        let encoded_message_id = ethabi::encode(values);
        let data: Vec<u8> = [function_signature.to_vec(), encoded_message_id].concat();

        let sdk_pub_key =
            secp256k1::PublicKey::from_bytes(env.signer().public_key().as_bytes()).unwrap();

        let caller = module_evm::derive_caller::from_sigspec(
            &SignatureAddressSpec::Secp256k1Eth(sdk_pub_key))
            .unwrap();

        let res: Vec<u8> = env
            .client()
            .query(
                env.client().latest_round().await?.into(),
                "evm.SimulateCall",
                module_evm::types::SimulateCallQuery {
                    gas_price: 10_000.into(),
                    gas_limit: 100_000,
                    caller,
                    address: Some(ORACLE_CONTRACT_ADDRESS.parse().unwrap()),
                    value: 0.into(),
                    data,
                },
            )
            .await?;

        Ok(res)
    }

    async fn get_unprocessed_messages(self: Arc<Self>, env: Environment<Self>) -> Result<Vec<u128>> {
        let res = self.make_query(
            env.clone(),
            "getProcessingMessages",
            &[], &[],
        ).await?;

        let decoded = ethabi::decode(
            &[ParamType::Array(Box::new(ParamType::Uint(256)))],
            &res,
        ).map_err(|e| anyhow::anyhow!("AIChatOracleApp - Failed to decode response: {}", e))?;

        // let value = decoded[0].clone().into_uint().unwrap().as_u128();
        let values = match &decoded[0] {
            Token::Array(tokens) => tokens,
            _ => return Err(anyhow::anyhow!("AIChatOracleApp - Decoded value is not an array")),
        };

        let uint_values: Vec<u128> = values
            .iter()
            .map(|token| token.clone().into_uint().unwrap().as_u128())
            .collect();

        Ok(uint_values)
    }

    async fn get_message_details(
        self: Arc<Self>,
        env: Environment<Self>,
        message_id: u128,
    ) -> Result<Message> {
        let res = self.make_query(
            env.clone(),
            "getMessage",
            &[ParamType::Uint(256)],
            &[Token::Uint(message_id.into())],
        ).await?;

        let decoded = ethabi::decode(&[
            ParamType::String, // _prompt
            ParamType::String, // _plugin
            ParamType::String, // _reply
        ], &res).map_err(|e| anyhow::anyhow!("AIChatOracleApp - Failed to decode response: {}", e))?;

        let message = Message {
            prompt: decoded[0].clone().into_string().unwrap(),
            plugin: decoded[1].clone().into_string().unwrap(),
            reply: decoded[2].clone().into_string().unwrap(),
        };

        Ok(message)
    }

    async fn get_agent_reply(self: Arc<Self>, message: Message) -> Result<String> {
        let encoded_prompt = encode(&message.prompt);
        let encoded_plugin = encode(&message.plugin);

        let api_url = format!("http://192.168.1.14:2266/Agent?prompt={}&plugin={}", encoded_prompt, encoded_plugin);

        let cfg = ureq::AgentConfig {
            https_only: false,
            user_agent: "rofl-utils/0.1.0".to_string(),
            ..Default::default()
        };
        let agent = rofl_utils::https::agent_with_config(cfg);
        let reply = tokio::task::spawn_blocking(move || -> Result<String> {
            let rsp: String = agent
                .get(api_url)
                .call()?
                .body_mut()
                .read_to_string()?;

            Ok(rsp)
        }).await??;

        Ok(reply)
    }

    async fn submit_agent_reply(
        &self,
        env: Environment<Self>,
        message_id: u128,
        reply: String,
    ) -> Result<()> {
        let function_signature = ethabi::short_signature("submitAgentReply", &[ParamType::Uint(256), ParamType::String]);
        let encoded_message_id = ethabi::encode(&[
            Token::Uint(message_id.into()),
            Token::String(reply),
        ]);
        let data: Vec<u8> = [function_signature.to_vec(), encoded_message_id].concat();

        let mut tx = self.new_transaction(
            "evm.Call",
            module_evm::types::Call {
                address: ORACLE_CONTRACT_ADDRESS.parse().unwrap(),
                value: 0.into(),
                data: data
            },
        );
        tx.set_fee_gas(200_000);
        env.client().sign_and_submit_tx(env.signer(), tx).await?;

        Ok(())
    }
}

fn main() {
    AIChatOracleApp.start();
}
