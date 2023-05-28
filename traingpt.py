import solcx
import torch
from transformers import GPT2Tokenizer, GPT2Model
import web3

# Solidity code
solidity_code = '''
pragma solidity ^0.8.0;

contract TrainingContract {
    // Define variables and data structures
    address[] public registeredNodes;
    string public sharedData;
    bytes[] public modelUpdates;
    string public aggregatedModel;
    
    // Node registration
    function registerNode() public {
        require(!isNodeRegistered(msg.sender), "Node is already registered");
        registeredNodes.push(msg.sender);
    }
    
    // Data sharing
    function shareData(string memory data) public {
        require(isNodeRegistered(msg.sender), "Node is not registered");
        sharedData = data;
    }
    
    // Model update submission
    function submitModelUpdate(bytes memory modelUpdate) public {
        require(isNodeRegistered(msg.sender), "Node is not registered");
        modelUpdates.push(modelUpdate);
    }
    
    // Model aggregation
    function aggregateModelUpdates() public {
        require(modelUpdates.length > 0, "No model updates available");
        
        bytes memory aggregated = new bytes(getTotalLength());
        uint256 currentIndex = 0;
        
        for (uint256 i = 0; i < modelUpdates.length; i++) {
            bytes memory update = modelUpdates[i];
            uint256 updateLength = update.length;
            
            for (uint256 j = 0; j < updateLength; j++) {
                aggregated[currentIndex] = update[j];
                currentIndex++;
            }
        }
        
        aggregatedModel = string(aggregated);
        
        delete modelUpdates;
    }
    
    // Helper function to check if a node is registered
    function isNodeRegistered(address node) private view returns (bool) {
        for (uint256 i = 0; i < registeredNodes.length; i++) {
            if (registeredNodes[i] == node) {
                return true;
            }
        }
        return false;
    }
    
    // Helper function to calculate the total length of all model updates
    function getTotalLength() private view returns (uint256) {
        uint256 totalLength = 0;
        for (uint256 i = 0; i < modelUpdates.length; i++) {
            totalLength += modelUpdates[i].length;
        }
        return totalLength;
    }
}

'''

ganache_url = "http://localhost:7545"
web3_instance = web3.Web3(web3.HTTPProvider(ganache_url))

# Step 2: Compile the Solidity code
compiled_contract = solcx.compile_source(solidity_code)
contract_abi = compiled_contract['<stdin>:TrainingContract']['abi']
contract_bytecode = compiled_contract['<stdin>:TrainingContract']['bin']

# Step 3: Deploy the Smart Contract
contract = web3_instance.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
contract_constructor = contract.constructor()
deployed_contract = contract_constructor.build_transaction({
    'from': web3_instance.eth.accounts[0],
    'gas': 20000000,  # Increase the gas limit to 20 million
    'gasPrice': web3_instance.to_wei('1', 'gwei')
})

transaction_hash = web3_instance.eth.send_transaction(deployed_contract)
transaction_receipt = web3_instance.eth.wait_for_transaction_receipt(transaction_hash)
contract_address = transaction_receipt['contractAddress']

# Step 4: Use the deployed contract for training
contract_instance = contract(address=contract_address)

# Step 5: Node Registration
node_address = "0x6Cd2F4868A2dD0dcEf2Bc51B8fc4b5Fcb40E2E9b"  # Address of the training node
contract_instance.functions.registerNode().transact({'from': node_address})

# Step 6: Data Distribution
with open("/media/revanthraja/Data/NLP_R/chat.txt", "r") as file:
    text_data = file.read()
contract_instance.functions.shareData(text_data).transact({'from': node_address})

# Step 7: Training Process
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

epochs = 10
for epoch in range(epochs):
    # Local model training using text_data
    inputs = tokenizer(text_data, add_special_tokens=True, return_tensors="pt")
    outputs = model(**inputs)
    model_update = outputs.logits

    # Step 8: Model Update
    model_update_str = model_update.tolist()  # Convert model update to string
    contract_instance.functions.submitModelUpdate(model_update_str).transact({'from': node_address})

# Step 9: Model Aggregation
contract_instance.functions.aggregateModelUpdates().transact({'from': node_address})

model = GPT2Model()
model.save_pretrained('trained_model')
