# NovaMart Multi-Agent Customer Support System

## Introduction
You are an Enterprise AI Solutions Architect at NovaMart, a fictional e-commerce company. The company's customer support team handles thousands of requests daily — order status inquiries, return and refund requests, policy questions, and more. Currently, support agents manually look up orders, cross-reference policies, and draft responses — a process that is slow, inconsistent, and error-prone.

Your job is to build the AI layer that automates this workflow.

---

## The Challenge
NovaMart needs a customer support system that can automatically understand a customer's request, gather relevant information from multiple data sources, make decisions based on company policies, and compose a professional response — all without human intervention. The system must handle multiple types of requests (order lookups, refund decisions, policy questions) by routing them to the right specialist, enforce enterprise safety guardrails, and provide full observability into every decision the AI makes.

---

## Your Product
You will build a production-grade multi-agent customer support system using **Amazon Bedrock AgentCore** and the **Strands Agents SDK**. The system consists of five AI agents working together in an **Orchestrator → Workers** hierarchy:

* **Orchestrator Agent**: Understands customer intent and routes requests to the right specialist
* **Inventory Agent**: Retrieves order and customer data from DynamoDB
* **Policy Agent**: Answers policy questions using parallel multi-agent RAG across three Knowledge Bases
* **Refund Agent**: Makes return eligibility decisions based on order data and policy rules
* **Communication Agent**: Composes the final customer-facing response

The system is deployed to AgentCore Runtime with Bedrock Guardrails for safety, session memory for multi-turn conversations, and CloudWatch/X-Ray observability for monitoring and debugging.

---

## Deliverables
To complete this project, you must submit:
1. **Completed `src/agent_orchestrator.py`** — All TODO items implemented (Tasks 2, 3, 4, 6)
2. **Three Bedrock Knowledge Bases** — Created in the AWS Console with synced data sources (Task 5)
3. **Populated `.env` file** — With your Knowledge Base IDs, AgentCore Runtime ARN, and Guardrail configuration
4. **Test Suite Verification Screenshot** — Showing all tests passed and a 120/120 score (`python tests/test_agent.py all`)
5. **X-Ray Service Map screenshot** — Showing the full trace of an end-to-end request through your multi-agent system (Task 6)

### Key Resources
* [Strands Agents SDK Documentation](https://strandsagents.com/) — Official documentation for the Strands Agents SDK used throughout the project — covering the `Agent` class, `@tool` decorator, and `BedrockModel` configuration.
* [Amazon Bedrock Knowledge Bases — Getting Started](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html) — Step-by-step AWS guide for creating Bedrock Knowledge Bases in the console — the exact steps required in Phase 3 of this project.
* [AgentCore CLI](https://github.com/aws/agentcore-cli) — The deployment tool used by this project. The supplied helper stages code, configures the runtime, and runs `agentcore deploy`. AWS SDK calls are still used for application services, guardrails, memory, and observability services.

---

## Project Environment & Foundation Infrastructure

Provided below is a Udacity workspace with Python and all required dependencies pre-installed. The workspace contains the starter project folder with everything you need to begin.

### Deploy the Foundation Infrastructure
The project includes a CloudFormation stack that provisions the AWS resources your agents will use. You need to deploy this stack before starting any implementation work.

The stack (`infrastructure/starter_stack.yaml`) creates:
* **DynamoDB tables**: `udacity-agentcore-orders`, `udacity-agentcore-customers`, `udacity-agentcore-workflow-state`
* **S3 bucket**: for policy documents. The AgentCore CLI stores its deployment package in the CDK bootstrap assets bucket, created on the first runtime deployment.
* **S3 Vectors**: a vector bucket with three vector indexes (`returns-policy-index`, `shipping-policy-index`, `warranty-policy-index`) — the backing store for the three Knowledge Bases
* **IAM role**: Execution role for the AgentCore Runtime with permissions for Bedrock, Knowledge Bases, DynamoDB, S3, S3 Vectors, CloudFormation, CloudWatch, and X-Ray
* **CloudWatch**: Log group for agent execution logs

#### Deploy the Stack
**Option A — AWS Console:**
1. Navigate to **AWS Console → CloudFormation → Create stack → With new resources (standard)**
2. Select **Upload a template file** → choose `infrastructure/starter_stack.yaml` → **Next**
3. Stack name: `udacity-agentcore` → **Next**
4. Leave stack options as defaults → **Next**
5. On the review page, check *"I acknowledge that AWS CloudFormation might create IAM resources with custom names"*
6. Click **Submit**
7. Wait for stack status to reach `CREATE_COMPLETE`

**Option B — AWS CloudShell:**
Open AWS Console → CloudShell (bottom-left icon), upload the template file, then run:
```bash
aws cloudformation deploy \
  --template-file starter_stack.yaml \
  --stack-name udacity-agentcore \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```
Check status:
```bash
aws cloudformation describe-stacks \
  --stack-name udacity-agentcore \
  --query "Stacks[0].StackStatus" \
  --region us-east-1
```
The stack is ready when the status shows `CREATE_COMPLETE`.

### Seed the Initial Data
After the stack is deployed, seed the DynamoDB tables and S3 bucket with sample data:
```bash
python infrastructure/seed_data.py
```
This populates:
* DynamoDB tables with mock customers (with Standard and Premium tiers) and orders
* S3 `policy-docs` bucket with sample return, shipping, and warranty policy documents

### Configure Environment Variables
```bash
cp .env.example .env
```
Most resource names are loaded automatically from CloudFormation exports — you don't need to fill them in manually. The `.env` file only holds values that are populated as you complete each task (Knowledge Base IDs, AgentCore Runtime ARN, Guardrail configuration).

### Verify Your Setup
```bash
python config.py
```
You should see a configuration table showing all resource names, including the S3 Vectors bucket and index names you will select in Phase 3. Fields showing `(not yet created)` are expected — they are populated as you complete each task.

### Local Machine Instructions
If you prefer to develop on your local machine instead of the workspace:
* Ensure you have Python 3.12+ and the AWS CLI configured with credentials for the `us-east-1` region
* Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Follow the same CloudFormation deployment and data seeding steps above.

---

## Architecture Overview

Your system follows an **Orchestrator → Workers** pattern where a central Orchestrator Agent routes customer requests to specialist worker agents. Each worker operates on a specific domain (inventory, policy, refunds, communication) and writes its findings to a shared DynamoDB `WorkflowState` record.

### Agent Graph
![Agent Graph](images/agent_graph.png)

*Figure 1: Customer request process managed by an OrchestratorAgent (Claude Haiku) routing requests and managing WorkflowState across specialist worker agents.*

---

### Request Flow
![Request Flow](images/request_flow.png)

*Figure 2: Sequence diagram illustrating how a typical return request flows from customer to Orchestrator, Inventory, Refund, Policy, and Communication Agents.*

---

### Shared WorkflowState
The `WorkflowState` record in DynamoDB uses **optimistic locking** — each update must include an `expected_version` that matches the current version. This prevents concurrent agents from overwriting each other's results.

![Shared WorkflowState](images/workflow_state.png)

*Figure 3: Session initialization and state progression from version 0 to version N across agent execution.*

---

### Recommended Agent Roles

| Agent | Model | Responsibility | Tools |
| :--- | :--- | :--- | :--- |
| **OrchestratorAgent** | Claude 4.5 Haiku | Routes requests, creates/updates WorkflowState | `initialize_session`, `route_to_inventory_agent`, `route_to_policy_agent`, `route_to_refund_agent`, `route_to_communication_agent` |
| **InventoryAgent** | Claude 4.5 Sonnet | Gathers order and customer facts from DynamoDB | `check_order_status`, `get_customer_tier`, `list_customer_orders` |
| **PolicyAgent** | Claude 4.5 Sonnet | Coordinates parallel RAG retrieval from 3 KBs, synthesizes results | `search_all_policies` (internally fans out to 3 retriever sub-agents) |
| **RefundAgent** | Claude 4.5 Sonnet | Makes return/refund eligibility decisions (30-day Standard, 60-day Premium) | `get_inventory_context`, `initiate_refund` |
| **CommunicationAgent** | Claude 4.5 Sonnet | Drafts final, empathetic customer-facing response | `get_full_workflow_context` |

---

## File Structure & Responsibilities

| File | Role | Action |
| :--- | :--- | :--- |
| `src/agent_orchestrator.py` | Main implementation file — agent builders, routing tools, deployment | Implement TODOs |
| `src/agent_utils.py` | Terminal trace UI, ANSI colours, `AgentTrace` class | Do not modify |
| `src/agent_observability.py` | X-Ray tracing (`@tool` wrapper) and CloudWatch logging | Do not modify |
| `src/bedrock_kb_retrieval.py` | KB retrieval helper used by PolicyAgent | Do not modify |
| `src/demo.py` | Runs one end-to-end request for local testing | Do not modify |
| `config.py` | Central configuration — reads CloudFormation exports + `.env` | Do not modify |
| `tests/test_agent.py` | Automated test suite | Do not modify |
| `src/agentcore_cli.py` | AgentCore CLI wrapper: stage code, configure, deploy, read runtime ARN | Do not modify |

### Supporting Modules Explained
* **`agent_utils.py`**: Contains the terminal trace UI that makes agent activity visible during chat and demo runs — colour-coded step banners, tool call formatting, parallel retrieval display, and the DynamoDB workflow summary.
* **`agent_observability.py`**: Provides the `@tool` decorator used in `agent_orchestrator.py` — wrapping the Strands `@tool` with an X-Ray subsegment per call, ensuring every request is traced to AWS X-Ray automatically. Use `@tool` exactly as you would with Strands.

### Strands Agents & WorkflowState Basics
* **Strands Agents SDK**: Build AI agents by decorating Python functions with `@tool` and passing them to an `Agent` along with a model and system prompt.
* **WorkflowState DynamoDB Helpers**:
  * `_create_workflow_state(session_id, customer_id)` — creates a blank record with version 0.
  * `_read_workflow_state(session_id)` — reads the current record and version.
  * `_update_workflow_state(session_id, updates, expected_version)` — writes with optimistic locking.

---

## Phase 1 / Task 2 — Implementing the Five Strands Agents

### 1.A — InventoryAgent
* **Role**: Gather order and customer facts from DynamoDB. **Never makes decisions** — only reports what it finds.
* **Tools**:
  * `check_order_status(customer_id: str, order_id: str)`: Looks up composite key in `config.ORDERS_TABLE`.
  * `get_customer_tier(customer_id: str)`: Looks up customer record in `config.CUSTOMERS_TABLE`.
  * `list_customer_orders(customer_id: str)`: Queries orders belonging to the customer.
* **Model**: `config.WORKER_MODEL_ID` (Claude Sonnet 4.5)
* **Temperature**: `0.1`

### 1.B — RefundAgent
* **Role**: Makes return/refund eligibility decisions based on facts in `WorkflowState` (populated by InventoryAgent) and tier policy windows.
* **Policy Windows**:
  * **Standard customers**: 30-day return window from delivery date.
  * **Premium customers**: 60-day return window from delivery date.
* **Tools**:
  * `get_inventory_context(session_id: str)`: Reads inventory findings from `WorkflowState`.
  * `initiate_refund(customer_id: str, order_id: str, reason: str)`: Updates order record status in DynamoDB to return/refund initiated.
* **Model**: `config.WORKER_MODEL_ID`
* **Temperature**: `0.1`

### 1.C — PolicyAgent — Multi-Agent RAG
* **Role**: Coordinator agent answering policy queries by fanning out to three specialized retriever sub-agents in parallel.
* **Retriever Sub-Agents (built inside `build_policy_agent()`)**:
  * `ReturnsPolicyRetrieverAgent`: Tool calls `retrieve_from_knowledge_base(query, config.RETURNS_KB_ID)`.
  * `ShippingPolicyRetrieverAgent`: Tool calls `retrieve_from_knowledge_base(query, config.SHIPPING_KB_ID)`.
  * `WarrantyPolicyRetrieverAgent`: Tool calls `retrieve_from_knowledge_base(query, config.WARRANTY_KB_ID)`.
  * Retriever Model: `config.WORKER_MODEL_ID`, Temperature: `0.0`
* **Coordinator Tool**:
  * `search_all_policies(query: str)`: Concurrently invokes all 3 retrievers using `concurrent.futures.ThreadPoolExecutor(max_workers=3)` and `as_completed()`, collecting and aggregating passages.
* **Coordinator Model**: `config.WORKER_MODEL_ID`, Temperature: `0.2`

### 1.D — CommunicationAgent
* **Role**: Drafts the final, customer-facing response. Employs a warm, empathetic, and professional tone.
* **Tools**:
  * `get_full_workflow_context(session_id: str)`: Reads the complete `WorkflowState` record containing inventory, refund, and policy results.
* **Model**: `config.WORKER_MODEL_ID`
* **Temperature**: `0.3`

### 1.E — OrchestratorAgent
* **Role**: High-level intent understanding and routing. **Never answers questions or writes customer responses directly** — delegates everything.
* **Routing Tools**:
  * `initialize_session(session_id: str, customer_id: str)`
  * `route_to_inventory_agent(session_id: str, query: str)`
  * `route_to_policy_agent(session_id: str, query: str)`
  * `route_to_refund_agent(session_id: str, query: str)`
  * `route_to_communication_agent(session_id: str, query: str)`
* **Routing Rules**:
  1. **Rule 1 (Every request, always)**: Call `initialize_session` first.
  2. **Rule 2 (Order status / return / refund requests)**: Route to Inventory Agent first, then Refund Agent.
  3. **Rule 3 (Policy meaning questions)**: (Return windows, shipping rates, warranty terms) → Route to Policy Agent.
  4. **Rule 4 (Account questions)**: ("what is my tier?", "am I premium?") → Route to Inventory Agent — never Policy Agent.
  5. **Rule 5 (Math / calculation questions)**: Orchestrator → CommunicationAgent, skipping Inventory, Policy, and Refund.
  6. **Rule 6 (Every request, always — last step)**: Route to Communication Agent to compose the final reply.
* **Model**: `config.ORCHESTRATOR_MODEL_ID` (Claude Haiku 4.5)
* **Temperature**: `0.0` (deterministic routing)

### Testing Task 2
```bash
python tests/test_agent.py task2
```
Live end-to-end testing options:
```bash
# Single pre-scripted request — runs one full refund scenario
python src/demo.py

# Runs 3 hardcoded scenarios and prints raw responses
python src/agent_orchestrator.py test

# Interactive terminal chat — type queries and watch the live trace
python src/agent_orchestrator.py chat
```

---

## Phase 2 / Task 3 & 4 — Guardrails, Runtime Deployment & Memory

### Task 3 — Guardrail and Runtime Deployment

#### `create_guardrail()`
Create a Bedrock Guardrail using `boto3.client('bedrock')`:
* **Content Policy**:
  * `SEXUAL`: `HIGH` strength
  * `VIOLENCE`: `HIGH` strength
  * `HATE`: `HIGH` strength
  * `INSULTS`: `MEDIUM` strength
  * `MISCONDUCT`: `MEDIUM` strength
* **PII Policy**:
  * `BLOCK`: Credit card numbers, Social Security Numbers (SSN)
  * `ANONYMIZE`: Email addresses, Phone numbers
* **Topic Policy**:
  * `DENY`: Competitor products, Pricing negotiations (haggling/altering advertised price; allow discount calculations), Legal threats
  * Configuration: `tierConfig = {"tierName": "STANDARD"}`, top-level `crossRegionConfig = {"guardrailProfileIdentifier": "us.guardrail.v1:0"}`
* **Word Policy**: Managed profanity list enabled
* **Messages**: Friendly blocked messages for both inputs and outputs
* **Versioning**: Publish a numbered version with `create_guardrail_version()` and return `(guardrail_id, guardrail_version)`. Never use `DRAFT`.

#### `deploy_to_agentcore_runtime()`
Stage and deploy using the supplied `agentcore_cli`:
1. Build runtime environment variables:
   * `AWS_REGION`
   * `PROJECT_NAME`
   * `RETURNS_KB_ID`
   * `SHIPPING_KB_ID`
   * `WARRANTY_KB_ID`
   * `AGENT_LOG_GROUP`
   * `GUARDRAIL_ID`
   * `GUARDRAIL_VERSION`
2. Pass to `agentcore_cli.configure_runtime()`:
   * `network_mode='PUBLIC'`
   * `protocol='HTTP'`
   * `execution_role_arn=config.AGENTCORE_ROLE_ARN`
3. Call `agentcore_cli.deploy()`
4. Retrieve ARN via `agentcore_cli.deployed_runtime_arn()` and return it.

### Task 4 — Memory Configuration

#### `configure_memory(runtime_arn)`
Configure session-scoped conversational context using AgentCore Memory:
* Call `agentcore_control.create_memory()`
* Strategy: `summaryMemoryStrategy` (`SESSION_SUMMARY`)
* Event Retention: `eventExpiryDuration = 7` (7 days)
* Wait for status to reach `ACTIVE`
* Return `memoryArn`

---

## Phase 3 / Task 5 — Bedrock Knowledge Bases (Console Setup)

Knowledge Bases are created in the AWS Console and powered by **S3 Vectors** and **Amazon Titan Embed Text v2**.

### Step 1: Verify Policy Documents
Policy documents were seeded under three S3 prefixes:
* `s3://{policy-docs-bucket}/policies/returns/`
* `s3://{policy-docs-bucket}/policies/shipping/`
* `s3://{policy-docs-bucket}/policies/warranty/`

Verify via AWS CloudShell or terminal:
```bash
aws s3 ls s3://$(python -c "import config; print(config.POLICY_BUCKET)")/policies/ --recursive
```

### Step 2: Create Three Self-Managed Knowledge Bases
In **AWS Console → Amazon Bedrock → Knowledge Bases → Create Knowledge Base**:

| Setting | Returns KB | Shipping KB | Warranty KB |
| :--- | :--- | :--- | :--- |
| **Name** | `novamart-returns-policy-kb` | `novamart-shipping-policy-kb` | `novamart-warranty-policy-kb` |
| **S3 Data Source Prefix** | `policies/returns/` | `policies/shipping/` | `policies/warranty/` |
| **Embedding Model** | Amazon Titan Embed Text v2 | Amazon Titan Embed Text v2 | Amazon Titan Embed Text v2 |
| **Vector Store** | S3 Vectors (Use existing bucket) | S3 Vectors (Use existing bucket) | S3 Vectors (Use existing bucket) |
| **S3 Vectors Bucket** | `Vector Bucket` from `python config.py` | Same | Same |
| **Vector Index** | `returns-policy-index` | `shipping-policy-index` | `warranty-policy-index` |

> [!CAUTION]
> Do NOT choose "Managed KB". Select "Self-managed KB" with the existing S3 Vectors bucket and indexes created by the CloudFormation stack.

**Important**: After creating each Knowledge Base, click **Sync** on the data source to index the policy documents.

### Step 3: Populate `.env`
Add the created KB IDs to `.env`:
```ini
RETURNS_KB_ID=your-returns-kb-id
SHIPPING_KB_ID=your-shipping-kb-id
WARRANTY_KB_ID=your-warranty-kb-id
```

### Test Task 5
```bash
python tests/test_agent.py task5
```

---

## Phase 4 / Task 6 — Observability & End-to-End Tracing

### `configure_observability(runtime_arn)`
Build logging and tracing configuration:
* **CloudWatch Logs (`cloudWatchConfig`)**:
  * Log Group: `config.AGENT_LOG_GROUP`
  * Log Level: `'INFO'`
  * `enabled = True`
* **X-Ray (`xRayConfig`)**:
  * `enabled = True`
  * `samplingRate = 1.0` (100% sampling)
* Pass to `apply_observability_config()` inside `try/except`.

### Test Task 6
```bash
python tests/test_agent.py task6
```

### Verification & X-Ray Service Map
Run a test request:
```bash
python src/agent_orchestrator.py test
```
Navigate to **AWS Console → CloudWatch → X-Ray traces → Service map** (or **X-Ray → Service map**), select **Last 5 minutes**. Verify the connected graph:
`NovaMart-Orchestrator` → Worker Agents (`InventoryAgent`, `RefundAgent`, `PolicyAgent`, `CommunicationAgent`) → `KnowledgeBase:*` nodes.

---

## Deployment & Submission Verification

### Full Deployment
```bash
python src/agent_orchestrator.py deploy
```

### End-to-End Scenarios
1. **Scenario 1 — Return Request**:
   * Input: `"I want to return my order ORD-27176"` (as `CUST-001`)
   * Routing: `Orchestrator → Inventory → Refund → Communication`
2. **Scenario 2 — Policy Inquiry**:
   * Input: `"What is the return policy for premium customers?"`
   * Routing: `Orchestrator → Policy (3 parallel KB retrievers) → Communication`
3. **Scenario 3 — Calculation**:
   * Input: `"How much are 5 items at $29.99 with 10% off?"`
   * Routing: `Orchestrator → CommunicationAgent` (skips Inventory, Policy, Refund)

### Full Test Suite Run
```bash
python tests/test_agent.py all
```
Ensure all tests pass and display a **120/120** score.

---

## Project Grading Rubric

### 1. Multi-Agent Graph
* **Worker Agents**:
  * `build_inventory_agent()` returns Strands Agent with `check_order_status`, `get_customer_tier`, `list_customer_orders`.
  * `build_refund_agent()` returns Strands Agent with `get_inventory_context`, `initiate_refund`. Standard window = 30 days, Premium = 60 days.
  * `build_communication_agent()` returns Strands Agent with `get_full_workflow_context`.
  * All workers use `config.WORKER_MODEL_ID` (Claude Sonnet 4.5).
  * Temperatures: Inventory (0.1), Refund (0.1), Communication (0.3).
  * Clear docstrings describing purpose, arguments, and return values.
* **Multi-Agent RAG**:
  * `build_policy_agent()` creates three retriever sub-agents: `ReturnsPolicyRetrieverAgent`, `ShippingPolicyRetrieverAgent`, `WarrantyPolicyRetrieverAgent`.
  * Each retriever uses `retrieve_from_knowledge_base()` with corresponding KB ID.
  * `search_all_policies` executes retrievers in parallel with `ThreadPoolExecutor(max_workers=3)` and `as_completed()`.
  * Retriever temperature = 0.0, coordinator temperature = 0.2.
* **Orchestrator Agent**:
  * Uses `config.ORCHESTRATOR_MODEL_ID` (Claude Haiku 4.5) with temperature 0.0.
  * Enforces all 6 routing rules.
  * Uses `initialize_session` first, updates `WorkflowState` with `expected_version`.
  * `route_to_communication_agent` is always the last tool call. Orchestrator never writes response itself.
  * `python tests/test_agent.py task2` passes.

### 2. AgentCore Runtime and Guardrails
* Bedrock Guardrail includes content, PII, topic, and word policies.
* Versioned guardrail published (`create_guardrail_version()`), returns `(guardrail_id, guardrail_version)`.
* Deployed via AgentCore CLI with public networking, HTTP, execution role, and required environment variables.
* Guardrail applied to agent models via `_apply_guardrail()`.
* `python tests/test_agent.py task3` passes.

### 3. Memory and Knowledge Bases
* `configure_memory()` sets up `summaryMemoryStrategy` with 7-day retention and returns `memoryArn`.
* 3 self-managed Bedrock Knowledge Bases created with Titan Embed Text v2 and S3 Vectors backing store.
* Synced data sources and populated `.env` IDs.
* `python tests/test_agent.py task4` and `task5` pass.

### 4. Observability
* `configure_observability()` configures CloudWatch Logs (INFO) and X-Ray (100% sampling).
* `python tests/test_agent.py task6` passes.
* X-Ray Service Map screenshot demonstrating full call chain.

### 5. Industry Best Practices
* Clean, modular code; snake_case naming; comprehensive docstrings.
* Self-contained agent builders returning exactly one Agent instance.
* Non-hardcoded configuration constants used throughout.

---

## Stand-Out Suggestions
1. **Adversarial Input Validation**: Run prompt injection, competitor mentions, and legal threats against the Guardrail and document blocked responses.
2. **Persistent Conversation Memory**: Add DynamoDB `agent-sessions` table with Strands SDK `DynamoDbSessionStorage` in Orchestrator to complement AgentCore cloud memory.
3. **CloudWatch Dashboard**: Build metrics for agent invocation counts, latency per agent, and Guardrail trip frequency.
4. **Cognito + Frontend UI**: Connect AWS Cognito and a React or HTML/JS web interface to the AgentCore Runtime endpoint.

---

## Submission Checklist
* [ ] Completed `src/agent_orchestrator.py`
* [ ] Populated `.env` file
* [ ] Screenshot of 120/120 test pass (`python tests/test_agent.py all`)
* [ ] Screenshot of X-Ray Service Map showing connected call chain
* [ ] Archive project files into a single `.zip` (under 500 MB) or provide public GitHub repo link

---

## Cleanup
After submission and evaluation, tear down resources to stop incurring charges:
```bash
# Dry run
python infrastructure/cleanup.py

# Delete resources
python infrastructure/cleanup.py --yes
```
