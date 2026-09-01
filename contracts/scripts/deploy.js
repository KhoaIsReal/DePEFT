/**
 * Hardhat / Node Deployment Script for DePEFT Smart Contracts
 * Deploys DePeftToken ($DEPEFT) and DePeftEscrow
 */
async function main() {
    console.log("=================================================");
    console.log(" Deploying DePEFT Network Smart Contracts       ");
    console.log("=================================================");

    const [deployer] = await ethers.getSigners();
    console.log("Deploying contracts with account:", deployer.address);

    // 1. Deploy DePeftToken ($DEPEFT) with 100,000,000 Initial Supply
    const initialSupply = 100_000_000;
    const DePeftToken = await ethers.getContractFactory("DePeftToken");
    const token = await DePeftToken.deploy(initialSupply);
    await token.waitForDeployment();
    const tokenAddress = await token.getAddress();
    console.log("✓ DePeftToken ($DEPEFT) deployed to:", tokenAddress);

    // 2. Deploy DePeftEscrow
    const DePeftEscrow = await ethers.getContractFactory("DePeftEscrow");
    const escrow = await DePeftEscrow.deploy(tokenAddress);
    await escrow.waitForDeployment();
    const escrowAddress = await escrow.getAddress();
    console.log("✓ DePeftEscrow deployed to:", escrowAddress);

    console.log("=================================================");
    console.log("Deployment complete! Contracts ready for Testnet/Mainnet.");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
