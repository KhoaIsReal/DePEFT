// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./DePeftToken.sol";

/**
 * @title DePeftEscrow
 * @notice On-chain bounty escrow and multi-round reward settlement contract for DePEFT.
 * @dev Locks client task deposits and releases payouts to tournament winners based on consensus proofs.
 */
contract DePeftEscrow {
    DePeftToken public immutable token;
    address public owner;

    struct TaskEscrow {
        address client;
        uint256 bountyPool;
        uint256 remainingBounty;
        uint256 totalRounds;
        uint256 completedRounds;
        bool isActive;
    }

    mapping(uint256 => TaskEscrow) public tasks;
    mapping(uint256 => mapping(uint256 => address)) public roundWinners;

    event TaskCreated(uint256 indexed taskId, address indexed client, uint256 bountyPool, uint256 totalRounds);
    event RoundSettled(uint256 indexed taskId, uint256 indexed round, address indexed winner, uint256 rewardAmount);
    event MultiRoundSettled(uint256 indexed taskId, uint256 indexed round, address[] winners, uint256[] rewardAmounts);
    event TaskRefunded(uint256 indexed taskId, address indexed client, uint256 refundAmount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only protocol owner authorized");
        _;
    }

    constructor(address _tokenAddress) {
        require(_tokenAddress != address(0), "Invalid token address");
        token = DePeftToken(_tokenAddress);
        owner = msg.sender;
    }

    /// Client deposits bounty tokens to fund a fine-tuning tournament task
    function depositTaskBounty(uint256 taskId, uint256 bountyAmount, uint256 totalRounds) external {
        require(tasks[taskId].client == address(0), "Task ID already exists");
        require(bountyAmount > 0, "Bounty must be greater than 0");
        require(totalRounds > 0, "Total rounds must be at least 1");

        // Transfer tokens from client to escrow
        require(token.transferFrom(msg.sender, address(this), bountyAmount), "Escrow transfer failed");

        tasks[taskId] = TaskEscrow({
            client: msg.sender,
            bountyPool: bountyAmount,
            remainingBounty: bountyAmount,
            totalRounds: totalRounds,
            completedRounds: 0,
            isActive: true
        });

        emit TaskCreated(taskId, msg.sender, bountyAmount, totalRounds);
    }

    /// Settle reward for a completed tournament round to winning miner
    function settleRoundReward(uint256 taskId, uint256 round, address winner) external onlyOwner {
        TaskEscrow storage task = tasks[taskId];
        require(task.isActive, "Task is not active");
        require(roundWinners[taskId][round] == address(0), "Round already settled");
        require(winner != address(0), "Invalid winner address");

        uint256 rewardPerRound;
        if (task.completedRounds + 1 >= task.totalRounds) {
            // Final round payout sweeps all remaining bounty to prevent integer division dust from being trapped
            rewardPerRound = task.remainingBounty;
            task.isActive = false;
        } else {
            rewardPerRound = task.bountyPool / task.totalRounds;
            require(task.remainingBounty >= rewardPerRound, "Insufficient remaining bounty");
        }

        task.remainingBounty -= rewardPerRound;
        task.completedRounds += 1;
        roundWinners[taskId][round] = winner;

        require(token.transfer(winner, rewardPerRound), "Reward payout transfer failed");
        emit RoundSettled(taskId, round, winner, rewardPerRound);
    }

    /// Settle reward for a completed tournament round to Top-K winning miners
    function settleRoundRewardTopK(
        uint256 taskId,
        uint256 round,
        address[] calldata winners,
        uint256[] calldata rewardAmounts
    ) external onlyOwner {
        require(winners.length > 0 && winners.length == rewardAmounts.length, "Invalid Top-K parameters");
        TaskEscrow storage task = tasks[taskId];
        require(task.isActive, "Task is not active");
        require(roundWinners[taskId][round] == address(0), "Round already settled");

        uint256 totalPayout = 0;
        for (uint256 i = 0; i < rewardAmounts.length; i++) {
            require(winners[i] != address(0), "Invalid winner address in Top-K");
            totalPayout += rewardAmounts[i];
        }

        require(task.remainingBounty >= totalPayout, "Insufficient remaining bounty for Top-K payout");

        task.remainingBounty -= totalPayout;
        task.completedRounds += 1;
        roundWinners[taskId][round] = winners[0]; // Set top-1 as primary round winner

        if (task.completedRounds >= task.totalRounds) {
            task.isActive = false;
        }

        for (uint256 i = 0; i < winners.length; i++) {
            if (rewardAmounts[i] > 0) {
                require(token.transfer(winners[i], rewardAmounts[i]), "Reward transfer failed in Top-K");
            }
        }

        emit MultiRoundSettled(taskId, round, winners, rewardAmounts);
    }

    /// Refund unspent bounty to client if task is cancelled
    function refundTask(uint256 taskId) external {
        TaskEscrow storage task = tasks[taskId];
        require(task.isActive, "Task is not active");
        require(msg.sender == task.client || msg.sender == owner, "Unauthorized refund request");

        uint256 refundAmount = task.remainingBounty;
        task.remainingBounty = 0;
        task.isActive = false;

        require(token.transfer(task.client, refundAmount), "Refund transfer failed");
        emit TaskRefunded(taskId, task.client, refundAmount);
    }
}
