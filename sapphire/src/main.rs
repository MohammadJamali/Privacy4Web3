use oasis_runtime_sdk::modules::rofl::app::prelude::*;

const ORACLE_CONTRACT_ADDRESS: &str = "ADDRESS";

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
        println!("ROFL chat service running...");
    }

    async fn on_runtime_block(self: Arc<Self>, env: Environment<Self>, _round: u64) {
        if let Err(err) = self.run_oracle(env).await {
            println!("Failed to check and process message: {:?}", err);
        }
    }
}

impl AIChatOracleApp {
    async fn run_oracle(self: Arc<Self>, env: Environment<Self>) -> Result<()> {
        println!("run_oracle");

        // let observation = tokio::task::spawn_blocking(move || -> Result<_> {
        //     // Request some data from Coingecko API.
        //     let rsp: serde_json::Value = rofl_utils::https::agent()
        //         .get("https://www.binance.com/api/v3/ticker/price?symbol=ROSEUSDT")
        //         .call()?
        //         .body_mut()
        //         .read_json()?;

        //     // Extract price and convert to integer.
        //     let price = rsp
        //         .pointer("/price")
        //         .ok_or(anyhow::anyhow!("price not available"))?
        //         .as_str().unwrap()
        //         .parse::<f64>()?;
        //     let price = (price * 1_000_000.0) as u128;

        //     Ok(price)
        // }).await??;


        // let mut tx = self.new_transaction(
        //     "evm.Call",
        //     module_evm::types::Call {
        //         address: ORACLE_CONTRACT_ADDRESS.parse().unwrap(),
        //         value: 0.into(),
        //         data: [
        //             ethabi::short_signature("submitObservation", &[ethabi::ParamType::Uint(128)])
        //                 .to_vec(),
        //             ethabi::encode(&[ethabi::Token::Uint(observation.into())]),
        //         ]
        //         .concat(),
        //     },
        // );
        // tx.set_fee_gas(200_000);

        // // Submit observation on chain.
        // env.client().sign_and_submit_tx(env.signer(), tx).await?;

        Ok(())
    }
}

fn main() {
    AIChatOracleApp.start();
}
