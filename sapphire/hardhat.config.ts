
import "@oasisprotocol/sapphire-hardhat"
import "@nomicfoundation/hardhat-toolbox";
import { HardhatUserConfig } from "hardhat/config";
import { HDAccountsUserConfig } from 'hardhat/types';

import "./tasks/deploy";

const TEST_HDWALLET: HDAccountsUserConfig = {
  mnemonic: 'test test test test test test test test test test test junk',
  path: "m/44'/60'/0'/0",
  initialIndex: 0,
  count: 20,
  passphrase: '',
};

const accounts = process.env.PRIVATE_KEY
  ? [process.env.PRIVATE_KEY]
  : TEST_HDWALLET;

const config: HardhatUserConfig = {
  solidity: {
    version: '0.8.20',
    settings: {
      optimizer: {
        enabled: true,
      },
    },
  },
  networks: {
    hardhat: {
      chainId: 1337
    },
    hardhat_local: {
      url: "http://localhost:8545",
    },
    'sapphire': {
      url: "https://sapphire.oasis.io",
      chainId: 0x5afe,
      accounts,
    },
    'sapphire-testnet': {
      url: "https://testnet.sapphire.oasis.io",
      chainId: 0x5aff,
      accounts,
    },
    'sapphire-localnet': {
      url: "http://localhost:8545",
      chainId: 0x5afd,
      accounts,
    },
  },
};

export default config;
