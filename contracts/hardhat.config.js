require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

const PRIVATE_KEY = process.env.PRIVATE_KEY || "0x0000000000000000000000000000000000000000000000000000000000000001";
const ETHERSCAN_API_KEY = process.env.ETHERSCAN_API_KEY || "";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    hardhat: {},
    localhost: {
      url: "http://127.0.0.1:8545",
    },
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "https://rpc.sepolia.org",
      accounts: [PRIVATE_KEY.startsWith("0x") ? PRIVATE_KEY : `0x${PRIVATE_KEY}`],
      chainId: 11155111,
    },
    baseSepolia: {
      url: process.env.BASE_SEPOLIA_RPC_URL || "https://sepolia.base.org",
      accounts: [PRIVATE_KEY.startsWith("0x") ? PRIVATE_KEY : `0x${PRIVATE_KEY}`],
      chainId: 84532,
    },
  },
  etherscan: {
    apiKey: ETHERSCAN_API_KEY,
  },
};
