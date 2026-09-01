// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title DePeftToken ($DEPEFT)
 * @notice Standard ERC-20 Utility & Governance Token for the DePEFT Network.
 * @dev Used for task bounty escrows, miner rewards, validator staking, and slashing.
 */
contract DePeftToken {
    string public name = "DePEFT Network Token";
    string public symbol = "DEPEFT";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    address public owner;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event TaskEscrowLocked(uint256 indexed taskId, address indexed client, uint256 amount);
    event RewardClaimed(uint256 indexed taskId, address indexed miner, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner authorized");
        _;
    }

    constructor(uint256 initialSupply) {
        owner = msg.sender;
        totalSupply = initialSupply * (10 ** uint256(decimals));
        balanceOf[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }

    function transfer(address to, uint256 value) public returns (bool success) {
        require(to != address(0), "Invalid recipient address");
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) public returns (bool success) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool success) {
        require(to != address(0), "Invalid recipient address");
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Allowance exceeded");
        balanceOf[from] -= value;
        allowance[from][msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(from, to, value);
        return true;
    }

    /// Faucet function for Testnet participants (capped per call)
    function testnetFaucet(address recipient, uint256 amount) external onlyOwner {
        require(balanceOf[owner] >= amount, "Faucet pool depleted");
        balanceOf[owner] -= amount;
        balanceOf[recipient] += amount;
        emit Transfer(owner, recipient, amount);
    }
}
