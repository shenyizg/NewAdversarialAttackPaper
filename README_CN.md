# Latest Adversarial Attack Papers
**update at 2025-08-04 09:18:57**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LeakyCLIP: Extracting Training Data from CLIP**

LeakyCLIP：从CLIP中提取训练数据 cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00756v1) [paper-pdf](http://arxiv.org/pdf/2508.00756v1)

**Authors**: Yunhao Chen, Shujie Wang, Xin Wang, Xingjun Ma

**Abstract**: Understanding the memorization and privacy leakage risks in Contrastive Language--Image Pretraining (CLIP) is critical for ensuring the security of multimodal models. Recent studies have demonstrated the feasibility of extracting sensitive training examples from diffusion models, with conditional diffusion models exhibiting a stronger tendency to memorize and leak information. In this work, we investigate data memorization and extraction risks in CLIP through the lens of CLIP inversion, a process that aims to reconstruct training images from text prompts. To this end, we introduce \textbf{LeakyCLIP}, a novel attack framework designed to achieve high-quality, semantically accurate image reconstruction from CLIP embeddings. We identify three key challenges in CLIP inversion: 1) non-robust features, 2) limited visual semantics in text embeddings, and 3) low reconstruction fidelity. To address these challenges, LeakyCLIP employs 1) adversarial fine-tuning to enhance optimization smoothness, 2) linear transformation-based embedding alignment, and 3) Stable Diffusion-based refinement to improve fidelity. Empirical results demonstrate the superiority of LeakyCLIP, achieving over 358% improvement in Structural Similarity Index Measure (SSIM) for ViT-B-16 compared to baseline methods on LAION-2B subset. Furthermore, we uncover a pervasive leakage risk, showing that training data membership can even be successfully inferred from the metrics of low-fidelity reconstructions. Our work introduces a practical method for CLIP inversion while offering novel insights into the nature and scope of privacy risks in multimodal models.

摘要: 了解对比语言-图像预训练（CLIP）中的记忆和隐私泄露风险对于确保多模式模型的安全性至关重要。最近的研究证明了从扩散模型中提取敏感训练示例的可行性，条件扩散模型表现出更强的记忆和泄露信息的倾向。在这项工作中，我们通过CLIP倒置的镜头调查了CLIP中的数据记忆和提取风险，这是一个旨在根据文本提示重建训练图像的过程。为此，我们引入了\textBF{LeakyCLIP}，这是一种新型攻击框架，旨在从CLIP嵌入中实现高质量、语义准确的图像重建。我们确定了CLIP倒置中的三个关键挑战：1）非鲁棒特征，2）文本嵌入中的视觉语义有限，以及3）重建保真度低。为了解决这些挑战，LeakyCLIP采用1）对抗性微调以增强优化平滑度，2）基于线性变换的嵌入对齐，以及3）基于稳定扩散的细化以提高保真度。经验结果证明了LeakyCLIP的优越性，与LAION-2B子集的基线方法相比，ViT-B-16的结构相似性指数测量（SSIM）提高了358%以上。此外，我们还发现了普遍存在的泄露风险，表明训练数据成员关系甚至可以从低保真重建的指标中成功推断出来。我们的工作介绍了一种实用的方法CLIP反演，同时提供了新的见解的性质和范围的隐私风险的多模态模型。



## **2. Criticality-Based Dynamic Topology Optimization for Enhancing Aerial-Marine Swarm Resilience**

基于临界度的动态布局优化提高航空-海洋群体韧性 cs.NI

Submit to INFOCOM 2026

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00688v1) [paper-pdf](http://arxiv.org/pdf/2508.00688v1)

**Authors**: Ruiyang Huang, Haocheng Wang, Yixuan Shen, Ning Gao, Qiang Ni, Shi Jin, Yifan Wu

**Abstract**: Heterogeneous marine-aerial swarm networks encounter substantial difficulties due to targeted communication disruptions and structural weaknesses in adversarial environments. This paper proposes a two-step framework to strengthen the network's resilience. Specifically, our framework combines the node prioritization based on criticality with multi-objective topology optimization. First, we design a three-layer architecture to represent structural, communication, and task dependencies of the swarm networks. Then, we introduce the SurBi-Ranking method, which utilizes graph convolutional networks, to dynamically evaluate and rank the criticality of nodes and edges in real time. Next, we apply the NSGA-III algorithm to optimize the network topology, aiming to balance communication efficiency, global connectivity, and mission success rate. Experiments demonstrate that compared to traditional methods like K-Shell, our SurBi-Ranking method identifies critical nodes and edges with greater accuracy, as deliberate attacks on these components cause more significant connectivity degradation. Furthermore, our optimization approach, when prioritizing SurBi-Ranked critical components under attack, reduces the natural connectivity degradation by around 30%, achieves higher mission success rates, and incurs lower communication reconfiguration costs, ensuring sustained connectivity and mission effectiveness across multi-phase operations.

摘要: 由于对抗环境中的有针对性的通信中断和结构性弱点，异类海空群网络遇到了巨大的困难。本文提出了一个两步框架来加强网络的弹性。具体来说，我们的框架将基于关键性的节点优先级与多目标布局优化相结合。首先，我们设计了一个三层架构来表示群网络的结构、通信和任务依赖性。然后，我们引入了SurBi-Ranking方法，该方法利用图卷积网络来实时动态评估和排名节点和边的关键性。接下来，我们应用NSGA-III算法来优化网络布局，旨在平衡通信效率、全球连接性和任务成功率。实验表明，与K-Shell等传统方法相比，我们的SurBi-Ranking方法以更高的准确性识别关键节点和边，因为对这些组件的故意攻击会导致更显着的连接性退化。此外，我们的优化方法在优先考虑受攻击的SurBi-Ranked关键组件时，将自然连接退化减少约30%，实现更高的任务成功率，并降低通信重新配置成本，确保多阶段操作中的持续连接和任务有效性。



## **3. Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights**

重新审视对象检测器上的对抗补丁防御：统一评估、大规模数据集和新见解 cs.CV

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00649v1) [paper-pdf](http://arxiv.org/pdf/2508.00649v1)

**Authors**: Junhao Zheng, Jiahao Sun, Chenhao Lin, Zhengyu Zhao, Chen Ma, Chong Zhang, Cong Wang, Qian Wang, Chao Shen

**Abstract**: Developing reliable defenses against patch attacks on object detectors has attracted increasing interest. However, we identify that existing defense evaluations lack a unified and comprehensive framework, resulting in inconsistent and incomplete assessments of current methods. To address this issue, we revisit 11 representative defenses and present the first patch defense benchmark, involving 2 attack goals, 13 patch attacks, 11 object detectors, and 4 diverse metrics. This leads to the large-scale adversarial patch dataset with 94 types of patches and 94,000 images. Our comprehensive analyses reveal new insights: (1) The difficulty in defending against naturalistic patches lies in the data distribution, rather than the commonly believed high frequencies. Our new dataset with diverse patch distributions can be used to improve existing defenses by 15.09% AP@0.5. (2) The average precision of the attacked object, rather than the commonly pursued patch detection accuracy, shows high consistency with defense performance. (3) Adaptive attacks can substantially bypass existing defenses, and defenses with complex/stochastic models or universal patch properties are relatively robust. We hope that our analyses will serve as guidance on properly evaluating patch attacks/defenses and advancing their design. Code and dataset are available at https://github.com/Gandolfczjh/APDE, where we will keep integrating new attacks/defenses.

摘要: 针对对象检测器上的补丁攻击开发可靠的防御吸引了越来越多的兴趣。然而，我们发现，现有的国防评估缺乏一个统一和全面的框架，导致目前的方法不一致和不完整的评估。为了解决这个问题，我们重新审视了11个代表性的防御，并提出了第一个补丁防御基准，涉及2个攻击目标，13个补丁攻击，11个对象检测器和4个不同的指标。这导致了具有94种类型的补丁和94，000张图像的大规模对抗补丁数据集。我们的全面分析揭示了新的见解：（1）防御自然主义补丁的困难在于数据分布，而不是普遍认为的高频。我们具有多样化补丁分布的新数据集可用于将现有防御改进15.09%AP@0.5。(2)受攻击对象的平均精度，而不是通常追求的补丁检测准确性，显示出与防御性能的高度一致性。(3)自适应攻击可以基本上绕过现有的防御，并且具有复杂/随机模型或通用补丁属性的防御相对稳健。我们希望我们的分析能够作为正确评估补丁攻击/防御并推进其设计的指导。代码和数据集可在https：//github.com/Gandolfczjh/APDE上获取，我们将在其中不断集成新的攻击/防御。



## **4. SwarnRaft: Leveraging Consensus for Robust Drone Swarm Coordination in GNSS-Degraded Environments**

SwarnRaft：利用共识在GNSS降级环境中实现稳健的无人机群协调 cs.DC

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00622v1) [paper-pdf](http://arxiv.org/pdf/2508.00622v1)

**Authors**: Kapel Dev, Yash Madhwal, Sofia Shevelo, Pavel Osinenko, Yury Yanovich

**Abstract**: Unmanned aerial vehicle (UAV) swarms are increasingly used in critical applications such as aerial mapping, environmental monitoring, and autonomous delivery. However, the reliability of these systems is highly dependent on uninterrupted access to the Global Navigation Satellite Systems (GNSS) signals, which can be disrupted in real-world scenarios due to interference, environmental conditions, or adversarial attacks, causing disorientation, collision risks, and mission failure. This paper proposes SwarnRaft, a blockchain-inspired positioning and consensus framework for maintaining coordination and data integrity in UAV swarms operating under GNSS-denied conditions. SwarnRaft leverages the Raft consensus algorithm to enable distributed drones (nodes) to agree on state updates such as location and heading, even in the absence of GNSS signals for one or more nodes. In our prototype, each node uses GNSS and local sensing, and communicates over WiFi in a simulated swarm. Upon signal loss, consensus is used to reconstruct or verify the position of the failed node based on its last known state and trajectory. Our system demonstrates robustness in maintaining swarm coherence and fault tolerance through a lightweight, scalable communication model. This work offers a practical and secure foundation for decentralized drone operation in unpredictable environments.

摘要: 无人机（UF）群越来越多地用于航空测绘、环境监测和自主交付等关键应用。然而，这些系统的可靠性高度依赖于不间断地访问全球导航卫星系统（GNSS）信号，而在现实世界场景中，这些信号可能会因干扰、环境条件或对抗性攻击而中断，从而导致方向迷失、碰撞风险和任务失败。本文提出了SwarnRaft，这是一个受区块链启发的定位和共识框架，用于维护在GNSS拒绝的条件下运行的无人机群的协调和数据完整性。SwarnRaft利用Raft共识算法，使分布式无人机（节点）能够就位置和航向等状态更新达成一致，即使在一个或多个节点没有GNSS信号的情况下也是如此。在我们的原型中，每个节点使用全球导航卫星系统和本地传感，并在模拟群中通过WiFi进行通信。信号丢失后，使用共识根据故障节点的最后已知状态和轨迹重建或验证故障节点的位置。我们的系统通过轻量级、可扩展的通信模型展示了在保持群体一致性和故障容忍方面的鲁棒性。这项工作为在不可预测的环境中去中心化的无人机操作提供了实用且安全的基础。



## **5. LeakSealer: A Semisupervised Defense for LLMs Against Prompt Injection and Leakage Attacks**

LeakSealer：LLM针对即时注入和泄漏攻击的半监督防御 cs.CR

22 pages, preprint

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00602v1) [paper-pdf](http://arxiv.org/pdf/2508.00602v1)

**Authors**: Francesco Panebianco, Stefano Bonfanti, Francesco Trovò, Michele Carminati

**Abstract**: The generalization capabilities of Large Language Models (LLMs) have led to their widespread deployment across various applications. However, this increased adoption has introduced several security threats, notably in the forms of jailbreaking and data leakage attacks. Additionally, Retrieval Augmented Generation (RAG), while enhancing context-awareness in LLM responses, has inadvertently introduced vulnerabilities that can result in the leakage of sensitive information. Our contributions are twofold. First, we introduce a methodology to analyze historical interaction data from an LLM system, enabling the generation of usage maps categorized by topics (including adversarial interactions). This approach further provides forensic insights for tracking the evolution of jailbreaking attack patterns. Second, we propose LeakSealer, a model-agnostic framework that combines static analysis for forensic insights with dynamic defenses in a Human-In-The-Loop (HITL) pipeline. This technique identifies topic groups and detects anomalous patterns, allowing for proactive defense mechanisms. We empirically evaluate LeakSealer under two scenarios: (1) jailbreak attempts, employing a public benchmark dataset, and (2) PII leakage, supported by a curated dataset of labeled LLM interactions. In the static setting, LeakSealer achieves the highest precision and recall on the ToxicChat dataset when identifying prompt injection. In the dynamic setting, PII leakage detection achieves an AUPRC of $0.97$, significantly outperforming baselines such as Llama Guard.

摘要: 大型语言模型（LLM）的概括能力导致它们在各种应用程序中广泛部署。然而，这种采用的增加引入了多种安全威胁，特别是越狱和数据泄露攻击的形式。此外，检索增强生成（RAG）虽然增强了LLM响应中的上下文感知，但无意中引入了可能导致敏感信息泄露的漏洞。我们的贡献是双重的。首先，我们介绍了一种方法来分析LLM系统的历史交互数据，从而生成按主题分类的使用地图（包括对抗性交互）。这种方法进一步为跟踪越狱攻击模式的演变提供了法医见解。其次，我们提出了LeakSealer，一个模型不可知的框架，它将静态分析与动态防御相结合，用于在人在回路（HITL）管道中进行取证洞察。这种技术可以识别主题组并检测异常模式，从而实现主动防御机制。我们在两种情况下经验评估LeakSealer：（1）越狱尝试，使用公共基准数据集，以及（2）由标记的LLM交互的精心策划数据集支持的PRI泄漏。在静态设置中，LeakSealer在识别提示注射时在ToxicChat数据集上实现了最高的精确度和召回率。在动态环境中，PIP泄漏检测的AUPRC为0.97美元，显着优于Llama Guard等基准。



## **6. ExclaveFL: Providing Transparency to Federated Learning using Exclaves**

ExclaveFL：使用Exclaves为联邦学习提供透明度 cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2412.10537v2) [paper-pdf](http://arxiv.org/pdf/2412.10537v2)

**Authors**: Jinnan Guo, Kapil Vaswani, Andrew Paverd, Peter Pietzuch

**Abstract**: In federated learning (FL), data providers jointly train a model without disclosing their training data. Despite its inherent privacy benefits, a malicious data provider can simply deviate from the correct training protocol without being detected, potentially compromising the trained model. While current solutions have explored the use of trusted execution environments (TEEs) to combat such attacks, they usually assume side-channel attacks against the TEEs are out of scope. However, such side-channel attacks can undermine the security properties of TEE-based FL frameworks, not by extracting the FL data, but by leaking keys that allow the adversary to impersonate as the TEE whilst deviating arbitrarily from the correct training protocol.   We describe ExclaveFL, an FL platform that provides end-to-end integrity and transparency, even in the presence of side-channel attacks on TEEs. We propose a new paradigm in which existing TEEs are used as exclaves -- integrity-protected execution environments that do not contain any secrets, making them immune to side-channel attacks. Whereas previous approaches attest the TEE itself and bind this attestation to a key held by the TEE, ExclaveFL attests individual data transformations at runtime. These runtime attestations form an attested dataflow graph, which can be checked to ensure the FL training job satisfies claims, such as deviations from the correct computation. We implement ExclaveFL by extending the popular NVFlare FL framework to use exclaves, and show experimentally that ExclaveFL introduces less than 10% overhead compared to the same FL framework without TEEs, whilst providing stronger security guarantees.

摘要: 在联合学习（FL）中，数据提供者联合训练模型，而不披露其训练数据。尽管恶意数据提供者具有固有的隐私优势，但它可能会在不被检测到的情况下偏离正确的训练协议，从而可能会损害训练后的模型。虽然当前的解决方案已经探索了使用可信执行环境（TEE）来对抗此类攻击，但它们通常认为针对TEE的侧通道攻击超出了范围。然而，此类侧通道攻击可能会破坏基于TE的FL框架的安全属性，而不是通过提取FL数据，而是通过泄露允许对手冒充TEK的密钥，同时任意偏离正确的训练协议。   我们描述了ExclaveFL，这是一个FL平台，即使在TEE上存在侧通道攻击的情况下也能提供端到端完整性和透明度。我们提出了一种新的范式，其中现有的TE被用作飞地--完整性保护的执行环境，不包含任何秘密，使其免受侧通道攻击。尽管以前的方法证明了TEK本身并将此证明绑定到TEK持有的密钥，但ExclaveFL在运行时证明了各个数据转换。这些运行时证明形成了经过证明的数据流图，可以检查该数据流图以确保FL培训作业满足声明，例如与正确计算的偏差。我们通过扩展流行的NVFlare FL框架来实现ExclaveFL，并通过实验表明，与没有TEE的相同FL框架相比，ExclaveFL引入了不到10%的开销，同时提供了更强的安全保证。



## **7. Wukong Framework for Not Safe For Work Detection in Text-to-Image systems**

文本到图像系统中工作不安全检测的悟空框架 cs.CV

Under review

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00591v1) [paper-pdf](http://arxiv.org/pdf/2508.00591v1)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Text-to-Image (T2I) generation is a popular AI-generated content (AIGC) technology enabling diverse and creative image synthesis. However, some outputs may contain Not Safe For Work (NSFW) content (e.g., violence), violating community guidelines. Detecting NSFW content efficiently and accurately, known as external safeguarding, is essential. Existing external safeguards fall into two types: text filters, which analyze user prompts but overlook T2I model-specific variations and are prone to adversarial attacks; and image filters, which analyze final generated images but are computationally costly and introduce latency. Diffusion models, the foundation of modern T2I systems like Stable Diffusion, generate images through iterative denoising using a U-Net architecture with ResNet and Transformer blocks. We observe that: (1) early denoising steps define the semantic layout of the image, and (2) cross-attention layers in U-Net are crucial for aligning text and image regions. Based on these insights, we propose Wukong, a transformer-based NSFW detection framework that leverages intermediate outputs from early denoising steps and reuses U-Net's pre-trained cross-attention parameters. Wukong operates within the diffusion process, enabling early detection without waiting for full image generation. We also introduce a new dataset containing prompts, seeds, and image-specific NSFW labels, and evaluate Wukong on this and two public benchmarks. Results show that Wukong significantly outperforms text-based safeguards and achieves comparable accuracy of image filters, while offering much greater efficiency.

摘要: 文本到图像（T2 I）生成是一种流行的AI生成内容（AIGC）技术，可实现多样化和创造性的图像合成。但是，某些输出可能包含不安全工作（NSFW）内容（例如，#21453;，违反了社会准则。有效和准确地检测NSFW内容，称为外部保护，是必不可少的。现有的外部保护措施分为两种类型：文本过滤器，它分析用户提示，但忽略T2 I模型特定的变化，并且容易受到对抗性攻击;图像过滤器，它分析最终生成的图像，但计算成本高昂并引入延迟。扩散模型是现代T2 I系统（如稳定扩散）的基础，它使用带有ResNet和Transformer模块的U-Net架构通过迭代去噪生成图像。我们观察到：（1）早期去噪步骤定义了图像的语义布局，（2）U-Net中的交叉注意层对于对齐文本和图像区域至关重要。基于这些见解，我们提出了Wukong，这是一个基于变换器的NSFW检测框架，它利用早期去噪步骤的中间输出并重用U-Net预训练的交叉注意力参数。悟空在扩散过程中运行，无需等待完整图像生成即可进行早期检测。我们还引入了一个包含提示、种子和特定于图像的NSFW标签的新数据集，并在这个基准和两个公共基准上评估悟空。结果表明，悟空的表现明显优于基于文本的保护措施，并实现了与图像过滤器相当的准确性，同时提供更高的效率。



## **8. Activation-Guided Local Editing for Jailbreaking Attacks**

越狱攻击的激活引导本地编辑 cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00555v1) [paper-pdf](http://arxiv.org/pdf/2508.00555v1)

**Authors**: Jiecong Wang, Haoran Li, Hao Peng, Ziqian Zeng, Zihao Wang, Haohua Du, Zhengtao Yu

**Abstract**: Jailbreaking is an essential adversarial technique for red-teaming these models to uncover and patch security flaws. However, existing jailbreak methods face significant drawbacks. Token-level jailbreak attacks often produce incoherent or unreadable inputs and exhibit poor transferability, while prompt-level attacks lack scalability and rely heavily on manual effort and human ingenuity. We propose a concise and effective two-stage framework that combines the advantages of these approaches. The first stage performs a scenario-based generation of context and rephrases the original malicious query to obscure its harmful intent. The second stage then utilizes information from the model's hidden states to guide fine-grained edits, effectively steering the model's internal representation of the input from a malicious toward a benign one. Extensive experiments demonstrate that this method achieves state-of-the-art Attack Success Rate, with gains of up to 37.74% over the strongest baseline, and exhibits excellent transferability to black-box models. Our analysis further demonstrates that AGILE maintains substantial effectiveness against prominent defense mechanisms, highlighting the limitations of current safeguards and providing valuable insights for future defense development. Our code is available at https://github.com/yunsaijc/AGILE.

摘要: 越狱是一项重要的对抗技术，可以将这些模型进行红色合作以发现和修补安全缺陷。然而，现有的越狱方法面临着重大缺陷。令牌级越狱攻击通常会产生不连贯或不可读的输入，并且表现出较差的可移植性，而预算级攻击缺乏可扩展性，并且严重依赖手动和人类聪明才智。我们提出了一个简洁有效的两阶段框架，结合了这些方法的优点。第一阶段执行基于情景的上下文生成并重新表达原始恶意查询以掩盖其有害意图。然后，第二阶段利用来自模型隐藏状态的信息来指导细粒度编辑，有效地引导模型对输入的内部表示从恶意转向良性。大量实验表明，该方法实现了最先进的攻击成功率，比最强基线的收益高达37.74%，并表现出出色的可移植性。我们的分析进一步表明，AGILE在对抗主要防御机制时保持了相当大的有效性，凸显了当前保障措施的局限性，并为未来的国防发展提供了宝贵的见解。我们的代码可在https://github.com/yunsaijc/AGILE上获取。



## **9. CyGATE: Game-Theoretic Cyber Attack-Defense Engine for Patch Strategy Optimization**

CyGATE：用于补丁策略优化的游戏理论网络攻击防御引擎 cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00478v1) [paper-pdf](http://arxiv.org/pdf/2508.00478v1)

**Authors**: Yuning Jiang, Nay Oo, Qiaoran Meng, Lu Lin, Dusit Niyato, Zehui Xiong, Hoon Wei Lim, Biplab Sikdar

**Abstract**: Modern cyber attacks unfold through multiple stages, requiring defenders to dynamically prioritize mitigations under uncertainty. While game-theoretic models capture attacker-defender interactions, existing approaches often rely on static assumptions and lack integration with real-time threat intelligence, limiting their adaptability. This paper presents CyGATE, a game-theoretic framework modeling attacker-defender interactions, using large language models (LLMs) with retrieval-augmented generation (RAG) to enhance tactic selection and patch prioritization. Applied to a two-agent scenario, CyGATE frames cyber conflicts as a partially observable stochastic game (POSG) across Cyber Kill Chain stages. Both agents use belief states to navigate uncertainty, with the attacker adapting tactics and the defender re-prioritizing patches based on evolving risks and observed adversary behavior. The framework's flexible architecture enables extension to multi-agent scenarios involving coordinated attackers, collaborative defenders, or complex enterprise environments with multiple stakeholders. Evaluated in a dynamic patch scheduling scenario, CyGATE effectively prioritizes high-risk vulnerabilities, enhancing adaptability through dynamic threat integration, strategic foresight by anticipating attacker moves under uncertainty, and efficiency by optimizing resource use.

摘要: 现代网络攻击分为多个阶段，要求防御者在不确定性下动态优先考虑缓解措施。虽然博弈论模型捕捉攻击者与防御者的相互作用，但现有方法通常依赖于静态假设，并且缺乏与实时威胁情报的集成，从而限制了其适应性。本文介绍了CyGATE，这是一个对攻击者-防御者交互进行建模的博弈论框架，使用大型语言模型（LLM）和检索增强生成（RAG）来增强策略选择和补丁优先级。CyGATE应用于双智能体场景，将网络冲突构建为跨网络杀戮链阶段的部分可观察随机游戏（POSG）。两个代理都使用信念状态来导航不确定性，攻击者调整策略，防御者根据不断变化的风险和观察到的对手行为重新确定补丁的优先级。该框架灵活的架构可以扩展到涉及协调攻击者，协作防御者或具有多个利益相关者的复杂企业环境的多代理场景。在动态补丁调度方案中进行评估，CyGATE有效地优先考虑高风险漏洞，通过动态威胁集成增强适应性，通过预测攻击者在不确定性下的行动增强战略远见，并通过优化资源使用提高效率。



## **10. System Identification from Partial Observations under Adversarial Attacks**

对抗性攻击下的部分观测结果识别系统 math.OC

8 pages, 3 figures

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2504.00244v2) [paper-pdf](http://arxiv.org/pdf/2504.00244v2)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper is concerned with the partially observed linear system identification, where the goal is to obtain reasonably accurate estimation of the balanced truncation of the true system up to order $k$ from output measurements. We consider the challenging case of system identification under adversarial attacks, where the probability of having an attack at each time is $\Theta(1/k)$ while the value of the attack is arbitrary. We first show that the $\ell_1$-norm estimator exactly identifies the true Markov parameter matrix for nilpotent systems under any type of attack. We then build on this result to extend it to general systems and show that the estimation error exponentially decays as $k$ grows. The estimated balanced truncation model accordingly shows an exponentially decaying error for the identification of the true system up to a similarity transformation. This work is the first to provide the input-output analysis of the system with partial observations under arbitrary attacks.

摘要: 本文涉及部分观测线性系统识别，目标是从输出测量中获得对高达k$阶的真实系统的平衡截断的相当准确的估计。我们考虑了对抗性攻击下系统识别的具有挑战性的情况，其中每次遭受攻击的概率为$\Theta（1/k）$，而攻击的值是任意的。我们首先表明，在任何类型的攻击下，$\ell_1 $-模估计器准确识别了幂零系统的真实Markov参数矩阵。然后，我们在这个结果的基础上将其扩展到一般系统，并表明估计误差随着$k$的增长而呈指数级衰减。因此，估计的平衡截断模型显示出指数衰减的误差，用于识别真实系统的相似变换。这项工作是第一个提供的输入输出分析系统的部分观测任意攻击。



## **11. Quantum Key-Recovery Attacks on FBC Algorithm**

对FBC算法的量子密钥恢复攻击 quant-ph

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00448v1) [paper-pdf](http://arxiv.org/pdf/2508.00448v1)

**Authors**: Yan-Ying Zhu, Bin-Bin Cai, Fei Gao, Song Lin

**Abstract**: With the advancement of quantum computing, symmetric cryptography faces new challenges from quantum attacks. These attacks are typically classified into two models: Q1 (classical queries) and Q2 (quantum superposition queries). In this context, we present a comprehensive security analysis of the FBC algorithm considering quantum adversaries with different query capabilities. In the Q2 model, we first design 4-round polynomial-time quantum distinguishers for FBC-F and FBC-KF structures, and then perform $r(r>6)$-round quantum key-recovery attacks. Our attacks require $O(2^{(2n(r-6)+3n)/2})$ quantum queries, reducing the time complexity by a factor of $2^{4.5n}$ compared with quantum brute-force search, where $n$ denotes the subkey length. Moreover, we give a new 6-round polynomial-time quantum distinguisher for FBC-FK structure. Based on this, we construct an $r(r>6)$-round quantum key-recovery attack with complexity $O(2^{n(r-6)})$. Considering an adversary with classical queries and quantum computing capabilities, we demonstrate low-data quantum key-recovery attacks on FBC-KF/FK structures in the Q1 model. These attacks require only a constant number of plaintext-ciphertext pairs, then use the Grover algorithm to search the intermediate states, thereby recovering all keys in $O(2^{n/2})$ time.

摘要: 随着量子计算的进步，对称密码学面临着量子攻击的新挑战。这些攻击通常分为两种模型：Q1（经典查询）和Q2（量子叠加查询）。在此背景下，我们考虑了具有不同查询能力的量子对手，对FBC算法进行了全面的安全分析。在Q2模型中，我们首先为FBC-F和FBC-KF结构设计4轮多次量子触发器，然后执行$r（r>6）$轮量子密钥恢复攻击。我们的攻击需要$O（2^{（2n（r-6）+3n）/2}）$量子查询，与量子暴力搜索相比，将时间复杂性降低了$2^{4.5n}$，其中$n$表示子密钥长度。此外，我们还给出了一种新的FBC-FK结构的6轮多次量子插值器。在此基础上，我们构造了一个复杂度为$O（2^{n（r-6）}）$的$r（r>6）$轮量子密钥恢复攻击。考虑到具有经典查询和量子计算能力的对手，我们在Q1模型中展示了对FBC-KF/FK结构的低数据量子密钥恢复攻击。这些攻击只需要固定数量的明文密文对，然后使用Grover算法搜索中间状态，从而在$O（2^{n/2}）$时间内恢复所有密钥。



## **12. Adaptive Branch Specialization in Spectral-Spatial Graph Neural Networks for Certified Robustness**

谱空间图神经网络的自适应分支专业化，以获得鲁棒性认证 cs.LG

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2505.08320v3) [paper-pdf](http://arxiv.org/pdf/2505.08320v3)

**Authors**: Yoonhyuk Choi, Jiho Choi, Chong-Kwon Kim

**Abstract**: Recent Graph Neural Networks (GNNs) combine spectral-spatial architectures for enhanced representation learning. However, limited attention has been paid to certified robustness, particularly regarding training strategies and underlying rationale. In this paper, we explicitly specialize each branch: the spectral network is trained to withstand l0 edge flips and capture homophilic structures, while the spatial part is designed to resist linf feature perturbations and heterophilic patterns. A context-aware gating network adaptively fuses the two representations, dynamically routing each node's prediction to the more reliable branch. This specialized adversarial training scheme uses branch-specific inner maximization (structure vs feature attacks) and a unified alignment objective. We provide theoretical guarantees: (i) expressivity of the gating mechanism beyond 1-WL, (ii) spectral-spatial frequency bias, and (iii) certified robustness with trade-off. Empirically, SpecSphere attains state-of-the-art node classification accuracy and offers tighter certified robustness on real-world benchmarks.

摘要: 最近的图神经网络（GNNs）结合了频谱空间架构，以增强表示学习。然而，对经认证的稳健性，特别是培训战略和基本原理的关注有限。在本文中，我们明确地专门化了每个分支：光谱网络被训练成能够承受l0边缘翻转并捕获同质结构，而空间部分被设计成能够抵抗linf特征扰动和异质模式。一个上下文感知的门控网络自适应地融合了这两种表示，动态地将每个节点的预测路由到更可靠的分支。这种专门的对抗训练方案使用特定于分支的内部最大化（结构与特征攻击）和统一的对齐目标。我们提供理论保证：（i）门控机制超过1-WL 1的表现力，（ii）频谱空间频率偏差，以及（iii）经过权衡的认证鲁棒性。从经验上看，SpecGlobe实现了最先进的节点分类准确性，并在现实世界基准上提供了更严格的认证稳健性。



## **13. Preliminary Investigation into Uncertainty-Aware Attack Stage Classification**

不确定性感知攻击阶段分类的初步研究 cs.CR

Proceedings for SPAIML2025 workshop, 26/10/2025 Bologna Italy,  co-located with ECAI2025

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00368v1) [paper-pdf](http://arxiv.org/pdf/2508.00368v1)

**Authors**: Alessandro Gaudenzi, Lorenzo Nodari, Lance Kaplan, Alessandra Russo, Murat Sensoy, Federico Cerutti

**Abstract**: Advanced Persistent Threats (APTs) represent a significant challenge in cybersecurity due to their prolonged, multi-stage nature and the sophistication of their operators. Traditional detection systems typically focus on identifying malicious activity in binary terms (benign or malicious) without accounting for the progression of an attack. However, effective response strategies depend on accurate inference of the attack's current stage, as countermeasures must be tailored to whether an adversary is in the early reconnaissance phase or actively conducting exploitation or exfiltration. This work addresses the problem of attack stage inference under uncertainty, with a focus on robustness to out-of-distribution (OOD) inputs. We propose a classification approach based on Evidential Deep Learning (EDL), which models predictive uncertainty by outputting parameters of a Dirichlet distribution over possible stages. This allows the system not only to predict the most likely stage of an attack but also to indicate when it is uncertain or the input lies outside the training distribution. Preliminary experiments in a simulated environment demonstrate that the proposed model can accurately infer the stage of an attack with calibrated confidence while effectively detecting OOD inputs, which may indicate changes in the attackers' tactics. These results support the feasibility of deploying uncertainty-aware models for staged threat detection in dynamic and adversarial environments.

摘要: 高级持续性威胁（APT）由于其长期、多阶段的性质及其运营商的复杂性，构成了网络安全的重大挑战。传统的检测系统通常专注于以二进制形式识别恶意活动（良性或恶意），而不考虑攻击的进展。然而，有效的应对策略取决于对攻击当前阶段的准确推断，因为应对措施必须根据对手是否处于早期侦察阶段或积极进行利用或撤离进行量身定制。这项工作解决了不确定性下的攻击阶段推理问题，重点关注对非分布（OOD）输入的鲁棒性。我们提出了一种基于证据深度学习（EDL）的分类方法，该方法通过输出可能阶段上的狄利克雷分布的参数来建模预测不确定性。这使得系统不仅能够预测攻击最有可能的阶段，而且还能够指示何时不确定或输入位于训练分布之外。模拟环境中的初步实验表明，所提出的模型可以以校准的置信度准确地推断攻击的阶段，同时有效地检测OOD输入，这可能表明攻击者策略的变化。这些结果支持在动态和对抗环境中部署不确定性感知模型进行分阶段威胁检测的可行性。



## **14. Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness**

通过最大化期望平坦度的低成本优化来提高对抗性可转移性 cs.CV

The original NCS method has been revised and renamed as MEF. A  theoretical proof of the relationship between flatness and transferability is  added

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2405.16181v2) [paper-pdf](http://arxiv.org/pdf/2405.16181v2)

**Authors**: Chunlin Qiu, Ang Li, Yiheng Duan, Shenyi Zhang, Yuanjie Zhang, Lingchen Zhao, Qian Wang

**Abstract**: Transfer-based attacks craft adversarial examples on white-box surrogate models and directly deploy them against black-box target models, offering model-agnostic and query-free threat scenarios. While flatness-enhanced methods have recently emerged to improve transferability by enhancing the loss surface flatness of adversarial examples, their divergent flatness definitions and heuristic attack designs suffer from unexamined optimization limitations and missing theoretical foundation, thus constraining their effectiveness and efficiency. This work exposes the severely imbalanced exploitation-exploration dynamics in flatness optimization, establishing the first theoretical foundation for flatness-based transferability and proposing a principled framework to overcome these optimization pitfalls. Specifically, we systematically unify fragmented flatness definitions across existing methods, revealing their imbalanced optimization limitations in over-exploration of sensitivity peaks or over-exploitation of local plateaus. To resolve these issues, we rigorously formalize average-case flatness and transferability gaps, proving that enhancing zeroth-order average-case flatness minimizes cross-model discrepancies. Building on this theory, we design a Maximin Expected Flatness (MEF) attack that enhances zeroth-order average-case flatness while balancing flatness exploration and exploitation. Extensive evaluations across 22 models and 24 current transfer-based attacks demonstrate MEF's superiority: it surpasses the state-of-the-art PGN attack by 4% in attack success rate at half the computational cost and achieves 8% higher success rate under the same budget. When combined with input augmentation, MEF attains 15% additional gains against defense-equipped models, establishing new robustness benchmarks. Our code is available at https://github.com/SignedQiu/MEFAttack.

摘要: 基于传输的攻击在白盒代理模型上制作对抗示例，并直接将它们部署到黑盒目标模型上，从而提供模型不可知且无需查询的威胁场景。虽然最近出现了平坦度增强方法，通过增强对抗示例的损失表面平坦度来提高可移植性，但它们不同的平坦度定义和启发式攻击设计存在未经审查的优化限制和缺乏理论基础，从而限制了它们的有效性和效率。这项工作揭示了平坦度优化中严重不平衡的开发-探索动态，为基于平坦度的可移植性奠定了第一个理论基础，并提出了克服这些优化陷阱的原则性框架。具体来说，我们系统地统一了现有方法中的碎片化平坦度定义，揭示了它们在过度探索灵敏度峰值或过度利用局部高原方面的不平衡优化局限性。为了解决这些问题，我们严格形式化了平均情况平坦度和可转移性差距，证明增强零阶平均情况平坦度可以最大限度地减少跨模型差异。在此理论的基础上，我们设计了一种最大期望平坦度（MEF）攻击，该攻击可以增强零阶平均情况平坦度，同时平衡平坦度探索和利用。对22个模型和24种当前基于传输的攻击进行了广泛评估，证明了MEF的优势：它的攻击成功率比最先进的PGN攻击高出4%，计算成本仅为一半，并在相同预算下实现了8%的成功率。与输入增强相结合时，MEF相对于配备防御装备的型号获得了15%的额外收益，从而建立了新的稳健性基准。我们的代码可在https://github.com/SignedQiu/MEFAttack上获取。



## **15. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

探索机器人学中视觉-语言-动作模型的对抗脆弱性 cs.RO

ICCV camera ready; Github:  https://github.com/William-wAng618/roboticAttack Homepage:  https://vlaattacker.github.io/

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2411.13587v4) [paper-pdf](http://arxiv.org/pdf/2411.13587v4)

**Authors**: Taowen Wang, Cheng Han, James Chenhao Liang, Wenhao Yang, Dongfang Liu, Luna Xinyu Zhang, Qifan Wang, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. Despite their significant capabilities, VLA models introduce new attack surfaces. This paper systematically evaluates their robustness. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce two untargeted attack objectives that leverage spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, we advance both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for continuously developing robust defense strategies prior to physical-world deployments.

摘要: 最近，在机器人领域，视觉-语言-动作（VLA）模型已成为一种变革性方法，使机器人能够通过在端到端学习框架内集成视觉和语言输入来执行复杂的任务。尽管VLA模型的功能很强，但它引入了新的攻击面。本文系统地评估了它们的稳健性。认识到机器人执行的独特需求，我们的攻击目标针对机器人系统固有的空间和功能特征。特别是，我们引入了两个利用空间基础来破坏机器人动作稳定的无针对性攻击目标，以及一个操纵机器人轨迹的有针对性攻击目标。此外，我们设计了一种对抗性补丁生成方法，可以在摄像机的视图中放置一个小的彩色补丁，从而在数字和物理环境中有效执行攻击。我们的评估显示，任务成功率显着下降，一系列模拟机器人任务的成功率下降高达100%，凸显了当前VLA架构中的关键安全漏洞。通过揭露这些漏洞并提出可操作的评估指标，我们促进了对基于VLA的机器人系统安全性的理解和增强，强调了在物理世界部署之前不断开发强大防御策略的必要性。



## **16. LLMs Encode Harmfulness and Refusal Separately**

法学硕士分别对有害和拒绝进行编码 cs.CL

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.11878v2) [paper-pdf](http://arxiv.org/pdf/2507.11878v2)

**Authors**: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi

**Abstract**: LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety

摘要: LLM接受过拒绝有害指令的培训，但他们真正了解除了拒绝之外的危害吗？先前的工作表明，LLM的拒绝行为可以通过一维子空间来调节，即拒绝方向。在这项工作中，我们确定了一个新的维度来分析LLM中的安全机制，即危害性，它在内部被编码为与拒绝分开的概念。存在一个与拒绝方向不同的危害方向。作为因果证据，沿着有害方向引导可能会导致LLM将无害的指令解释为有害的，但沿着拒绝方向引导往往会直接引发拒绝反应，而不会扭转模型对有害性的判断。此外，使用我们确定的危害性概念，我们发现某些越狱方法通过减少拒绝信号来发挥作用，而不会扭转模型内部的危害性信念。我们还发现，对抗性地微调模型以接受有害的指令对模型的内部有害信念的影响最小。这些见解导致了一个实际的安全应用：该模型的潜在危害表示可以作为一个内在的保障（潜在的警卫），用于检测不安全的输入和减少过度拒绝，这是强大的微调攻击。例如，我们的Latent Guard在不同的越狱方法中实现了与Llama Guard 3 8B相当或更好的性能，Llama Guard 3 8B是一种专用的微调保护模型。我们的研究结果表明，LLM对危害性的内部理解比他们拒绝不同输入指令的决定更强大，为研究AI安全性提供了一个新的视角



## **17. DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing**

DCT-Shield：针对恶意图像编辑的稳健频域防御 cs.CV

Accepted to ICCV 2025

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2504.17894v2) [paper-pdf](http://arxiv.org/pdf/2504.17894v2)

**Authors**: Aniruddha Bala, Rohit Chowdhury, Rohan Jaiswal, Siddharth Roheda

**Abstract**: Advancements in diffusion models have enabled effortless image editing via text prompts, raising concerns about image security. Attackers with access to user images can exploit these tools for malicious edits. Recent defenses attempt to protect images by adding a limited noise in the pixel space to disrupt the functioning of diffusion-based editing models. However, the adversarial noise added by previous methods is easily noticeable to the human eye. Moreover, most of these methods are not robust to purification techniques like JPEG compression under a feasible pixel budget. We propose a novel optimization approach that introduces adversarial perturbations directly in the frequency domain by modifying the Discrete Cosine Transform (DCT) coefficients of the input image. By leveraging the JPEG pipeline, our method generates adversarial images that effectively prevent malicious image editing. Extensive experiments across a variety of tasks and datasets demonstrate that our approach introduces fewer visual artifacts while maintaining similar levels of edit protection and robustness to noise purification techniques.

摘要: 扩散模型的进步使得通过文本提示轻松编辑图像成为可能，这引起了人们对图像安全性的担忧。可以访问用户图像的攻击者可以利用这些工具进行恶意编辑。最近的防御尝试通过在像素空间中添加有限的噪声来破坏基于扩散的编辑模型的功能来保护图像。然而，由先前方法添加的对抗性噪声很容易被人眼注意到。此外，这些方法中的大多数对于在可行的像素预算下的JPEG压缩等净化技术并不鲁棒。我们提出了一种新颖的优化方法，通过修改输入图像的离散Cosine变换（离散Cosine变换）系数，直接在频域中引入对抗性扰动。通过利用JPEG管道，我们的方法生成对抗图像，可以有效防止恶意图像编辑。针对各种任务和数据集的广泛实验表明，我们的方法引入了更少的视觉伪影，同时保持了类似水平的编辑保护和对噪音净化技术的鲁棒性。



## **18. Beyond Optimal Fault Tolerance**

超越最佳故障容忍度 cs.DC

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2501.06044v7) [paper-pdf](http://arxiv.org/pdf/2501.06044v7)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.

摘要: 任何协议都可以实现的最佳故障容差在各种设置中都有其特征。例如，对于在部分同步设置中运行的状态机复制（SVR）协议，可以同时保证针对$\Alpha $-有界对手（即，控制少于参与者$\Alpha $一部分的对手）和针对$\Beta $的活力-有界的对手当且仅当$\Alpha +2\Beta\leq 1 $。   本文描述了当放宽标准一致性要求以允许有界数量$r $的一致性违规时，SVR协议在多大程度上可能实现“优于最优”的公差保证。我们证明，如果没有额外的时间假设，限制回滚是不可能的，并研究每当攻击时间左右的消息延迟受到参数$\Delta '*$（该参数可以任意大于部分同步模型中限制后GST消息延迟的参数$\Delta $）时，能够容忍一致性违规并从一致性违规中恢复的协议。在这里，协议的故障容限可以是$r $的非常函数，并且我们证明，对于每个$r $，任何SVR协议可实现的最佳“可恢复故障容限”的上下限和下限匹配。例如，对于在部分同步设置中保证针对1/3有界对手的活性的协议，5/9有界对手总是会导致一次一致性违规，但不会导致两次一致性违规，而2/3有界对手总是会导致两次一致性违规，但不会导致三次。我们的积极结果是通过通用的“恢复程序”实现的，该程序可以移植到任何负责任的SVR协议上，并在违规后恢复一致性，同时仅回滚在之前$2\Delta '*$时间步中完成的事务。



## **19. Graph Representation-based Model Poisoning on Federated Large Language Models**

基于图表示的模型对联邦大型语言模型的毒害 cs.CR

7 pages, 5 figures (Submitted to IEEE Communication Magazine)

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.01694v2) [paper-pdf](http://arxiv.org/pdf/2507.01694v2)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) enable powerful generative capabilities within wireless networks while preserving data privacy. Nonetheless, FedLLMs remain vulnerable to model poisoning attacks. This article first reviews recent advancements in model poisoning techniques and existing defense mechanisms for FedLLMs, underscoring critical limitations, especially when dealing with non-IID textual data distributions. Current defense strategies predominantly employ distance or similarity-based outlier detection mechanisms, relying on the assumption that malicious updates markedly differ from benign statistical patterns. However, this assumption becomes inadequate against adaptive adversaries targeting billion-parameter LLMs. The article further investigates graph representation-based model poisoning (GRMP), an emerging attack paradigm that exploits higher-order correlations among benign client gradients to craft malicious updates indistinguishable from legitimate ones. GRMP can effectively circumvent advanced defense systems, causing substantial degradation in model accuracy and overall performance. Moreover, the article outlines a forward-looking research roadmap that emphasizes the necessity of graph-aware secure aggregation methods, specialized vulnerability metrics tailored for FedLLMs, and evaluation frameworks to enhance the robustness of federated language model deployments.

摘要: 联合大型语言模型（FedLLM）在无线网络中实现强大的生成能力，同时保护数据隐私。尽管如此，FedLLM仍然容易受到模型中毒攻击。本文首先回顾了FedLLM模型中毒技术和现有防御机制的最新进展，强调了关键局限性，尤其是在处理非IID文本数据分布时。当前的防御策略主要采用基于距离或相似性的离群值检测机制，其基础是恶意更新与良性统计模式显着不同的假设。然而，对于针对数十亿参数LLM的自适应对手来说，这一假设变得不充分。本文进一步研究了基于图表示的模型中毒（GRMP），这是一种新兴的攻击范式，利用良性客户端梯度之间的更高层相关性来制作与合法更新无法区分的恶意更新。GRMP可以有效规避先进的防御系统，导致模型准确性和整体性能大幅下降。此外，本文概述了一个前瞻性的研究路线图，强调图形感知的安全聚合方法、为FedLLM量身定制的专业漏洞指标以及增强联邦语言模型部署稳健性的评估框架的必要性。



## **20. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

多模态LLM越狱的概率建模：从量化到应用 cs.CR

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2503.06989v2) [paper-pdf](http://arxiv.org/pdf/2503.06989v2)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 最近，多模式大型语言模型（MLLM）展示了其在理解多模式内容方面的卓越能力。然而，它们仍然容易受到越狱攻击，这些攻击利用其安全调整中的弱点来产生有害反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLM的能力的二元分类是不合适的。从这个观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表当提示此输入时MLLM生成恶意响应的可能性。我们通过对MLLM的多次查询来估算这一可能性。使用越狱概率预测网络（JPPN）对输入隐藏状态与其相应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体来说，我们提出了基于越狱概率的攻击（JPA），该攻击优化输入图像上的对抗性扰动以最大化越狱概率，并通过包括单调文本改写进一步将其增强为多模式JPA（MJPA）。为了对抗攻击，我们还提出了基于越狱概率的微调（JPF），它通过MLLM参数更新最大限度地降低越狱概率。大量实验表明，（1）（M）JPA在白盒和黑匣子设置下攻击广泛的模型时都能产生显着的改进。(2)JPF最多将越狱人数大幅减少60%以上。上述两个结果都证明了引入越狱概率以在输入越狱能力之间进行细微差别的重要性。



## **21. Scalable and Precise Patch Robustness Certification for Deep Learning Models with Top-k Predictions**

具有Top-k预测的深度学习模型的可扩展且精确的补丁鲁棒性认证 cs.LG

accepted by QRS 2025

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.23335v1) [paper-pdf](http://arxiv.org/pdf/2507.23335v1)

**Authors**: Qilin Zhou, Haipeng Wang, Zhengyuan Wei, W. K. Chan

**Abstract**: Patch robustness certification is an emerging verification approach for defending against adversarial patch attacks with provable guarantees for deep learning systems. Certified recovery techniques guarantee the prediction of the sole true label of a certified sample. However, existing techniques, if applicable to top-k predictions, commonly conduct pairwise comparisons on those votes between labels, failing to certify the sole true label within the top k prediction labels precisely due to the inflation on the number of votes controlled by the attacker (i.e., attack budget); yet enumerating all combinations of vote allocation suffers from the combinatorial explosion problem. We propose CostCert, a novel, scalable, and precise voting-based certified recovery defender. CostCert verifies the true label of a sample within the top k predictions without pairwise comparisons and combinatorial explosion through a novel design: whether the attack budget on the sample is infeasible to cover the smallest total additional votes on top of the votes uncontrollable by the attacker to exclude the true labels from the top k prediction labels. Experiments show that CostCert significantly outperforms the current state-of-the-art defender PatchGuard, such as retaining up to 57.3% in certified accuracy when the patch size is 96, whereas PatchGuard has already dropped to zero.

摘要: 补丁稳健性认证是一种新兴的验证方法，用于防御对抗性补丁攻击，并为深度学习系统提供可证明的保证。经过认证的回收技术可以保证预测经过认证的样本的唯一真实标签。然而，如果适用于前k预测，现有技术通常对标签之间的这些选票进行成对比较，由于攻击者控制的选票数量的膨胀，无法准确地证明前k预测标签内的唯一真实标签（即，攻击预算）;然而，列举投票分配的所有组合会遇到组合爆炸问题。我们提出CostCert，这是一种新颖的、可扩展的、精确的基于投票的认证恢复捍卫者。CostCert通过新颖的设计验证前k个预测中样本的真实标签，无需成对比较和组合爆炸：对样本的攻击预算是否不可行，无法覆盖攻击者无法控制的投票之上的最小总额外投票，以将真实标签从前k个预测标签中排除。实验表明，CostCert的表现显着优于当前最先进的防御器PatchGuard，例如当补丁大小为96时，认证准确率可保持高达57.3%，而PatchGuard已经降至零。



## **22. Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems via Knowledge Asymmetry Exploitation**

通过知识不对称利用从检索增强生成系统中进行细粒度隐私提取 cs.CR

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.23229v1) [paper-pdf](http://arxiv.org/pdf/2507.23229v1)

**Authors**: Yufei Chen, Yao Wang, Haibin Zhang, Tao Gu

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge bases, but this advancement introduces significant privacy risks. Existing privacy attacks on RAG systems can trigger data leakage but often fail to accurately isolate knowledge-base-derived sentences within mixed responses. They also lack robustness when applied across multiple domains. This paper addresses these challenges by presenting a novel black-box attack framework that exploits knowledge asymmetry between RAG and standard LLMs to achieve fine-grained privacy extraction across heterogeneous knowledge landscapes. We propose a chain-of-thought reasoning strategy that creates adaptive prompts to steer RAG systems away from sensitive content. Specifically, we first decompose adversarial queries to maximize information disparity and then apply a semantic relationship scoring to resolve lexical and syntactic ambiguities. We finally train a neural network on these feature scores to precisely identify sentences containing private information. Unlike prior work, our framework generalizes to unseen domains through iterative refinement without pre-defined knowledge. Experimental results show that we achieve over 91% privacy extraction rate in single-domain and 83% in multi-domain scenarios, reducing sensitive sentence exposure by over 65% in case studies. This work bridges the gap between attack and defense in RAG systems, enabling precise extraction of private information while providing a foundation for adaptive mitigation.

摘要: 检索增强生成（RAG）系统通过集成外部知识库来增强大型语言模型（LLM），但这一进步带来了巨大的隐私风险。对RAG系统的现有隐私攻击可能会引发数据泄露，但通常无法准确地隔离混合响应中的知识库派生句子。当应用于多个领域时，它们也缺乏稳健性。本文通过提出一种新型的黑匣子攻击框架来解决这些挑战，该框架利用RAG和标准LLM之间的知识不对称性来实现跨异类知识环境的细粒度隐私提取。我们提出了一种思想链推理策略，可以创建自适应提示来引导RAG系统远离敏感内容。具体来说，我们首先分解对抗性查询以最大化信息差异，然后应用语义关系评分来解决词汇和语法歧义。我们最终根据这些特征分数训练神经网络，以精确识别包含私人信息的句子。与之前的工作不同，我们的框架通过迭代细化而无需预先定义的知识，将其推广到不可见的领域。实验结果表明，我们在单域场景中实现了超过91%的隐私提取率，在多域场景中实现了83%的隐私提取率，在案例研究中将敏感句子暴露减少了超过65%。这项工作弥合了RAG系统中攻击和防御之间的差距，能够精确提取私人信息，同时为自适应缓解提供基础。



## **23. Adversarial-Guided Diffusion for Multimodal LLM Attacks**

多模式LLM攻击的对抗引导扩散 cs.CV

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.23202v1) [paper-pdf](http://arxiv.org/pdf/2507.23202v1)

**Authors**: Chengwei Xia, Fan Ma, Ruijie Quan, Kun Zhan, Yi Yang

**Abstract**: This paper addresses the challenge of generating adversarial image using a diffusion model to deceive multimodal large language models (MLLMs) into generating the targeted responses, while avoiding significant distortion of the clean image. To address the above challenges, we propose an adversarial-guided diffusion (AGD) approach for adversarial attack MLLMs. We introduce adversarial-guided noise to ensure attack efficacy. A key observation in our design is that, unlike most traditional adversarial attacks which embed high-frequency perturbations directly into the clean image, AGD injects target semantics into the noise component of the reverse diffusion. Since the added noise in a diffusion model spans the entire frequency spectrum, the adversarial signal embedded within it also inherits this full-spectrum property. Importantly, during reverse diffusion, the adversarial image is formed as a linear combination of the clean image and the noise. Thus, when applying defenses such as a simple low-pass filtering, which act independently on each component, the adversarial image within the noise component is less likely to be suppressed, as it is not confined to the high-frequency band. This makes AGD inherently robust to variety defenses. Extensive experiments demonstrate that our AGD outperforms state-of-the-art methods in attack performance as well as in model robustness to some defenses.

摘要: 本文解决了使用扩散模型来欺骗多模式大型语言模型（MLLM）生成有针对性的响应，同时避免干净图像的显着失真来生成对抗图像的挑战。为了解决上述挑战，我们提出了一种针对对抗攻击MLLM的对抗引导扩散（AGD）方法。我们引入对抗引导噪音以确保攻击效果。我们设计中的一个关键观察是，与大多数将高频扰动直接嵌入到干净图像中的传统对抗攻击不同，AGD将目标语义注入到反向扩散的噪音成分中。由于扩散模型中添加的噪音跨越整个频谱，因此嵌入其中的对抗信号也继承了这种全频谱属性。重要的是，在反向扩散期间，对抗图像形成为干净图像和噪音的线性组合。因此，当应用独立作用于每个分量的简单低通过滤等防御措施时，噪音分量内的对抗图像不太可能被抑制，因为它不限于高频带。这使得AGD对多样性防御具有内在的鲁棒性。大量实验表明，我们的AGD在攻击性能以及对某些防御的模型鲁棒性方面优于最先进的方法。



## **24. AUV-Fusion: Cross-Modal Adversarial Fusion of User Interactions and Visual Perturbations Against VARS**

AUV-Fusion：针对VARS的用户交互和视觉扰动的跨模态对抗性融合 cs.IR

14 pages,6 figures

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22880v1) [paper-pdf](http://arxiv.org/pdf/2507.22880v1)

**Authors**: Hai Ling, Tianchi Wang, Xiaohao Liu, Zhulin Tao, Lifang Yang, Xianglin Huang

**Abstract**: Modern Visual-Aware Recommender Systems (VARS) exploit the integration of user interaction data and visual features to deliver personalized recommendations with high precision. However, their robustness against adversarial attacks remains largely underexplored, posing significant risks to system reliability and security. Existing attack strategies suffer from notable limitations: shilling attacks are costly and detectable, and visual-only perturbations often fail to align with user preferences. To address these challenges, we propose AUV-Fusion, a cross-modal adversarial attack framework that adopts high-order user preference modeling and cross-modal adversary generation. Specifically, we obtain robust user embeddings through multi-hop user-item interactions and transform them via an MLP into semantically aligned perturbations. These perturbations are injected onto the latent space of a pre-trained VAE within the diffusion model. By synergistically integrating genuine user interaction data with visually plausible perturbations, AUV-Fusion eliminates the need for injecting fake user profiles and effectively mitigates the challenge of insufficient user preference extraction inherent in traditional visual-only attacks. Comprehensive evaluations on diverse VARS architectures and real-world datasets demonstrate that AUV-Fusion significantly enhances the exposure of target (cold-start) items compared to conventional baseline methods. Moreover, AUV-Fusion maintains exceptional stealth under rigorous scrutiny.

摘要: 现代视觉感知推荐系统（VAR）利用用户交互数据和视觉特征的集成来提供高精度的个性化推荐。然而，它们对对抗攻击的鲁棒性在很大程度上仍未得到充分开发，这对系统的可靠性和安全性构成了重大风险。现有的攻击策略存在明显的局限性：先令攻击成本高昂且可检测，并且仅视觉干扰通常无法与用户偏好保持一致。为了应对这些挑战，我们提出了AUV-Fusion，这是一种跨模式对抗攻击框架，采用高级用户偏好建模和跨模式对手生成。具体来说，我们通过多跳用户项交互来获得稳健的用户嵌入，并通过MLP将它们转换为语义对齐的扰动。这些扰动被注入到扩散模型内预训练的VAE的潜在空间中。通过将真实的用户交互数据与视觉上合理的干扰协同集成，AUV-Fusion消除了注入虚假用户配置文件的需要，并有效地缓解了传统纯视觉攻击中固有的用户偏好提取不足的挑战。对各种VAR架构和现实世界数据集的全面评估表明，与传统基线方法相比，AUV-Fusion显着增强了目标（冷启动）物品的暴露。此外，AUV-Fusion在严格审查下保持了出色的隐身性。



## **25. Curvature Dynamic Black-box Attack: revisiting adversarial robustness via dynamic curvature estimation**

弯曲动态黑匣子攻击：通过动态弯曲估计重新审视对抗鲁棒性 cs.LG

This article contains several flaws

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2505.19194v2) [paper-pdf](http://arxiv.org/pdf/2505.19194v2)

**Authors**: Peiran Sun

**Abstract**: Adversarial attack reveals the vulnerability of deep learning models. For about a decade, countless attack and defense methods have been proposed, leading to robustified classifiers and better understanding of models. Among these methods, curvature-based approaches have attracted attention because it is assumed that high curvature may give rise to rough decision boundary. However, the most commonly used \textit{curvature} is the curvature of loss function, scores or other parameters from within the model as opposed to decision boundary curvature, since the former can be relatively easily formed using second order derivative. In this paper, we propose a new query-efficient method, dynamic curvature estimation(DCE), to estimate the decision boundary curvature in a black-box setting. Our approach is based on CGBA, a black-box adversarial attack. By performing DCE on a wide range of classifiers, we discovered, statistically, a connection between decision boundary curvature and adversarial robustness. We also propose a new attack method, curvature dynamic black-box attack(CDBA) with improved performance using the dynamically estimated curvature.

摘要: 对抗性攻击揭示了深度学习模型的脆弱性。大约十年来，人们提出了无数的攻击和防御方法，从而产生了鲁棒化分类器并更好地理解模型。在这些方法中，基于弯曲的方法引起了人们的关注，因为人们认为高弯曲可能会产生粗略的决策边界。然而，最常用的\textit{currency}是模型内的损失函数、分数或其他参数的弯曲，而不是决策边界弯曲，因为前者可以相对容易地使用二阶求导形成。在本文中，我们提出了一种新的查询高效方法--动态弯曲估计（VCE），来估计黑匣子环境下的决策边界弯曲。我们的方法基于CGBA，这是一种黑匣子对抗攻击。通过对广泛的分类器执行VCE，我们从统计上发现了决策边界弯曲和对抗鲁棒性之间的联系。我们还提出了一种新的攻击方法：弯曲动态黑匣子攻击（CDBA），使用动态估计的弯曲来提高性能。



## **26. DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion**

Distill：通过潜在扩散对可疑特洛伊木马输入进行无数据倒置 cs.CV

ICCV 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22813v1) [paper-pdf](http://arxiv.org/pdf/2507.22813v1)

**Authors**: Hossein Mirzaei, Zeinab Taghavi, Sepehr Rezaee, Masoud Hadi, Moein Madadi, Mackenzie W. Mathis

**Abstract**: Deep neural networks have demonstrated remarkable success across numerous tasks, yet they remain vulnerable to Trojan (backdoor) attacks, raising serious concerns about their safety in real-world mission-critical applications. A common countermeasure is trigger inversion -- reconstructing malicious "shortcut" patterns (triggers) inserted by an adversary during training. Current trigger-inversion methods typically search the full pixel space under specific assumptions but offer no assurances that the estimated trigger is more than an adversarial perturbation that flips the model output. Here, we propose a data-free, zero-shot trigger-inversion strategy that restricts the search space while avoiding strong assumptions on trigger appearance. Specifically, we incorporate a diffusion-based generator guided by the target classifier; through iterative generation, we produce candidate triggers that align with the internal representations the model relies on for malicious behavior. Empirical evaluations, both quantitative and qualitative, show that our approach reconstructs triggers that effectively distinguish clean versus Trojaned models. DISTIL surpasses alternative methods by high margins, achieving up to 7.1% higher accuracy on the BackdoorBench dataset and a 9.4% improvement on trojaned object detection model scanning, offering a promising new direction for reliable backdoor defense without reliance on extensive data or strong prior assumptions about triggers. The code is available at https://github.com/AdaptiveMotorControlLab/DISTIL.

摘要: 深度神经网络在众多任务中取得了显着的成功，但它们仍然容易受到特洛伊木马（后门）攻击，这引发了人们对其在现实世界任务关键型应用中安全性的严重担忧。常见的对策是触发倒置--重建对手在训练期间插入的恶意“快捷”模式（触发器）。当前的触发器倒置方法通常在特定假设下搜索整个像素空间，但不能保证估计的触发不仅仅是翻转模型输出的对抗性扰动。在这里，我们提出了一种无数据、零触发触发器倒置策略，该策略限制了搜索空间，同时避免了对触发器外观的强烈假设。具体来说，我们结合了一个由目标分类器引导的基于扩散的生成器;通过迭代生成，我们生成与模型所依赖的恶意行为的内部表示一致的候选触发器。定量和定性的经验评估表明，我们的方法重建了触发器，可以有效区分干净模型和特洛伊模型。Distill以很高的优势超越了替代方法，在BackdoorBench数据集上实现了高达7.1%的准确性，在木马对象检测模型扫描上提高了9.4%，为可靠的后门防御提供了一个有希望的新方向，而无需依赖大量数据或对触发器的强大先验假设。该代码可在https://github.com/AdaptiveMotorControlLab/DISTIL上获取。



## **27. Cryptanalysis of LC-MUME: A Lightweight Certificateless Multi-User Matchmaking Encryption for Mobile Devices**

LC-MUME的加密分析：一种用于移动设备的轻量级无证书多用户匹配加密 cs.CR

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22674v1) [paper-pdf](http://arxiv.org/pdf/2507.22674v1)

**Authors**: Ramprasad Sarkar

**Abstract**: Yang et al. proposed a lightweight certificateless multiuser matchmaking encryption (LC-MUME) scheme for mobile devices, published in IEEE Transactions on Information Forensics and Security (TIFS) (DOI: 10.1109/TIFS.2023.3321961). Their construction aims to reduce computational and communication overhead within a one-to-many certificateless cryptographic framework. The authors claim that their scheme satisfies existential unforgeability under chosen-message attacks (EUF-CMA) in the random oracle model. However, our cryptanalytic study demonstrates that the scheme fails to meet this critical security requirement. In particular, we show that a Type-I adversary can successfully forge a valid ciphertext without possessing the complete private key of the sender. Both theoretical analysis and practical implementation confirm that this attack can be mounted with minimal computational cost. To address these weaknesses, we propose a modification strategy to strengthen the security of matchmaking encryption schemes in mobile computing environments.

摘要: Yang等人提出了一种针对移动设备的轻量级无证书多用户匹配加密（LC-MUME）方案，发表在《IEEE信息取证与安全交易》（TIFS）（DOI：10.1109/TIFS.2023.3321961）中。他们的构建旨在减少一对多无证书加密框架内的计算和通信负担。作者声称，他们的方案在随机预言模型中满足选择消息攻击（EUF-CMA）下的存在不可伪造性。然而，我们的密码分析研究表明，该计划未能满足这一关键的安全要求。特别是，我们表明I型对手可以在不拥有发送者完整的私有密钥的情况下成功伪造有效的密文。理论分析和实际实现都证实，这种攻击可以以最小的计算成本发起。为了解决这些弱点，我们提出了一种修改策略来加强移动计算环境中匹配加密方案的安全性。



## **28. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

不要落后，RAG：使用RAG进行免训练对抗检测 cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.04858v3) [paper-pdf](http://arxiv.org/pdf/2504.04858v3)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.

摘要: 对抗性补丁攻击通过嵌入误导深度模型的局部扰动对视觉系统构成重大威胁。传统的防御方法通常需要重新训练或微调，这使得它们在实际部署中不切实际。我们提出了一个免训练的视觉检索增强生成（VRAG）框架，该框架集成了视觉语言模型（VLM）用于对抗性补丁检测。通过在不断扩展的数据库中检索与存储的攻击相似的视觉上相似的补丁和图像，VRAG执行生成推理以识别不同的攻击类型，所有这些都无需额外的训练或微调。我们广泛评估了开源大型VLM，包括Qwen-VL-Plus、Qwen2.5-VL-72 B和UI-TARS-72 B-DPO，以及Gemini-2.0（一种闭源模型）。值得注意的是，开源UI-TARS-72 B-DPO模型实现了高达95%的分类准确率，为开源对抗补丁检测奠定了新的最新水平。Gemini-2.0的总体准确率达到了最高的98%，但仍然是闭源的。实验结果证明了VRAG在以最少的人类注释识别各种对抗补丁方面的有效性，为针对不断发展的对抗补丁攻击的稳健、实用的防御铺平了道路。



## **29. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

基于扩散的对抗性身份操纵用于面部隐私保护 cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.21646v3) [paper-pdf](http://arxiv.org/pdf/2504.21646v3)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.

摘要: 由于社交网络上潜在的未经授权的监视和用户跟踪，面部识别（FR）系统的成功引发了严重的隐私问题。现有的增强隐私的方法无法生成可以保护面部隐私的自然面部图像。在本文中，我们提出了基于扩散的对抗身份操纵（DiffAIM）来生成针对恶意FR系统的自然且高度可转移的对抗面孔。具体来说，我们在扩散模型的低维潜在空间内操纵面部身份。这涉及在反向扩散过程中迭代地注入基于梯度的对抗性身份指导，逐步引导一代人走向所需的对抗性面孔。该指南针对向目标的身份融合进行了优化，同时促进源自源头的语义分歧，促进有效模仿，同时保持视觉自然性。我们进一步结合了结构保留的正规化，以在操作过程中保持面部结构一致性。针对人脸验证和识别任务的大量实验表明，与最新技术相比，迪夫AIM实现了更强的黑匣子攻击可转移性，同时保持了卓越的视觉质量。我们还证明了所提出的方法对商业FR API（包括Face++和Aliyun）的有效性。



## **30. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

利用协同认知偏见来绕过LLC的安全性 cs.CL

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22564v1) [paper-pdf](http://arxiv.org/pdf/2507.22564v1)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.

摘要: 大型语言模型（LLM）在广泛的任务中表现出令人印象深刻的能力，但它们的安全机制仍然容易受到利用认知偏见（系统性偏离理性判断）的对抗攻击。与之前专注于即时工程或算法操纵的越狱方法不同，这项工作强调了多偏差相互作用在破坏LLM保障措施方面被忽视的力量。我们提出了CognitiveAttack，这是一种新型的红色团队框架，可以系统地利用个人和组合的认知偏见。通过集成有监督的微调和强化学习，CognitiveAttack生成嵌入优化的偏差组合的提示，有效地绕过安全协议，同时保持高攻击成功率。实验结果揭示了30种不同的LLM存在重大漏洞，特别是在开源模型中。与SOTA黑匣子方法PAP相比，CognitiveAttack的攻击成功率高得多（60.1% vs 31.6%），暴露了当前防御机制的严重局限性。这些发现凸显了多偏见相互作用是一种强大但未充分探索的攻击载体。这项工作通过连接认知科学和LLM安全性，引入了一种新颖的跨学科视角，为更强大、更人性化的人工智能系统铺平了道路。



## **31. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

使用具有指定概率操纵的白盒对抗攻击对DNN模型进行所有权验证 cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2505.17579v3) [paper-pdf](http://arxiv.org/pdf/2505.17579v3)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Isobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.

摘要: 在本文中，我们提出了一种新的框架，用于图像分类任务的深度神经网络（DNN）模型的所有权验证。它允许合法所有者和第三方验证型号身份，而无需出示原始型号。我们假设一个灰盒场景，其中未经授权的用户拥有从原始模型非法复制的模型，在云环境中提供服务，用户抛出图像并接收分类结果作为输出类的概率分布。该框架应用白盒对抗攻击，将特定类的输出概率与指定值对齐。由于对原始模型的了解，它使所有者能够生成此类对抗性示例。我们通过引入控制参数，提出了一种基于迭代快速梯度符号法（FGSM）的简单但有效的对抗攻击方法。实验结果证实了使用对抗攻击识别DNN模型的有效性。



## **32. RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function**

RCR-AF：通过Rademacher复杂性降低激活函数增强模型概括 cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22446v1) [paper-pdf](http://arxiv.org/pdf/2507.22446v1)

**Authors**: Yunrui Yu, Kafeng Wang, Hang Su, Jun Zhu

**Abstract**: Despite their widespread success, deep neural networks remain critically vulnerable to adversarial attacks, posing significant risks in safety-sensitive applications. This paper investigates activation functions as a crucial yet underexplored component for enhancing model robustness. We propose a Rademacher Complexity Reduction Activation Function (RCR-AF), a novel activation function designed to improve both generalization and adversarial resilience. RCR-AF uniquely combines the advantages of GELU (including smoothness, gradient stability, and negative information retention) with ReLU's desirable monotonicity, while simultaneously controlling both model sparsity and capacity through built-in clipping mechanisms governed by two hyperparameters, $\alpha$ and $\gamma$. Our theoretical analysis, grounded in Rademacher complexity, demonstrates that these parameters directly modulate the model's Rademacher complexity, offering a principled approach to enhance robustness. Comprehensive empirical evaluations show that RCR-AF consistently outperforms widely-used alternatives (ReLU, GELU, and Swish) in both clean accuracy under standard training and in adversarial robustness within adversarial training paradigms.

摘要: 尽管深度神经网络取得了广泛成功，但仍然极易受到对抗攻击，从而在安全敏感的应用中构成了重大风险。本文研究了激活函数作为增强模型稳健性的关键但未充分研究的组件。我们提出了Rademacher复杂性降低激活函数（RCR-AF），这是一种新型激活函数，旨在提高概括性和对抗韧性。RCR-AF独特地结合了GELU的优势（包括平滑性、梯度稳定性和负信息保留性）与ReLU理想的单调性，同时通过由两个超参数$\Alpha$和$\gamma$管理的内置剪裁机制控制模型稀疏性和容量。我们基于Rademacher复杂性的理论分析表明，这些参数直接调节模型的Rademacher复杂性，提供了一种增强稳健性的原则方法。全面的实证评估表明，RCR-AF在标准训练下的清晰准确性和对抗训练范式内的对抗鲁棒性方面始终优于广泛使用的替代方案（ReLU、GELU和Swish）。



## **33. Theoretical Analysis of Relative Errors in Gradient Computations for Adversarial Attacks with CE Loss**

具有CE损失的对抗性攻击梯度计算相对误差的理论分析 cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22428v1) [paper-pdf](http://arxiv.org/pdf/2507.22428v1)

**Authors**: Yunrui Yu, Hang Su, Cheng-zhong Xu, Zhizhong Su, Jun Zhu

**Abstract**: Gradient-based adversarial attacks using the Cross-Entropy (CE) loss often suffer from overestimation due to relative errors in gradient computation induced by floating-point arithmetic. This paper provides a rigorous theoretical analysis of these errors, conducting the first comprehensive study of floating-point computation errors in gradient-based attacks across four distinct scenarios: (i) unsuccessful untargeted attacks, (ii) successful untargeted attacks, (iii) unsuccessful targeted attacks, and (iv) successful targeted attacks. We establish theoretical foundations characterizing the behavior of relative numerical errors under different attack conditions, revealing previously unknown patterns in gradient computation instability, and identify floating-point underflow and rounding as key contributors. Building on this insight, we propose the Theoretical MIFPE (T-MIFPE) loss function, which incorporates an optimal scaling factor $T = t^*$ to minimize the impact of floating-point errors, thereby enhancing the accuracy of gradient computation in adversarial attacks. Extensive experiments on the MNIST, CIFAR-10, and CIFAR-100 datasets demonstrate that T-MIFPE outperforms existing loss functions, including CE, C\&W, DLR, and MIFPE, in terms of attack potency and robustness evaluation accuracy.

摘要: 由于浮点算法引起的梯度计算中的相对误差，使用交叉Entropy（CE）损失的基于对象的对抗攻击经常会被高估。本文对这些错误进行了严格的理论分析，首次对四种不同场景下基于梯度的攻击中的浮点计算错误进行了全面研究：（i）不成功的非目标攻击，（ii）成功的非目标攻击，（iii）不成功的目标攻击，（iv）成功的目标攻击。我们建立了描述不同攻击条件下相对数值误差行为的理论基础，揭示了梯度计算不稳定性中以前未知的模式，并确定浮点下溢和舍入是关键因素。基于这一见解，我们提出了理论MIFPE（T-MIFPE）损失函数，它结合了最佳缩放因子$T = t^*$，以最大限度地减少浮点错误的影响，从而提高对抗性攻击中梯度计算的准确性。对MNIST、CIFAR-10和CIFAR-100数据集的广泛实验表明，T-MIFPE在攻击效力和鲁棒性评估准确性方面优于现有的损失函数，包括CE、C & W、DLR和MIFPE。



## **34. Benchmarking Fraud Detectors on Private Graph Data**

针对私有图表数据对欺诈检测器进行基准测试 cs.CR

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22347v1) [paper-pdf](http://arxiv.org/pdf/2507.22347v1)

**Authors**: Alexander Goldberg, Giulia Fanti, Nihar Shah, Zhiwei Steven Wu

**Abstract**: We introduce the novel problem of benchmarking fraud detectors on private graph-structured data. Currently, many types of fraud are managed in part by automated detection algorithms that operate over graphs. We consider the scenario where a data holder wishes to outsource development of fraud detectors to third parties (e.g., vendors or researchers). The third parties submit their fraud detectors to the data holder, who evaluates these algorithms on a private dataset and then publicly communicates the results. We propose a realistic privacy attack on this system that allows an adversary to de-anonymize individuals' data based only on the evaluation results. In simulations of a privacy-sensitive benchmark for facial recognition algorithms by the National Institute of Standards and Technology (NIST), our attack achieves near perfect accuracy in identifying whether individuals' data is present in a private dataset, with a True Positive Rate of 0.98 at a False Positive Rate of 0.00. We then study how to benchmark algorithms while satisfying a formal differential privacy (DP) guarantee. We empirically evaluate two classes of solutions: subsample-and-aggregate and DP synthetic graph data. We demonstrate through extensive experiments that current approaches do not provide utility when guaranteeing DP. Our results indicate that the error arising from DP trades off between bias from distorting graph structure and variance from adding random noise. Current methods lie on different points along this bias-variance trade-off, but more complex methods tend to require high-variance noise addition, undermining utility.

摘要: 我们引入了在私人图形结构数据上对欺诈检测器进行基准测试的新颖问题。目前，许多类型的欺诈在一定程度上是通过图形操作的自动检测算法来管理的。我们考虑数据持有者希望将欺诈检测器的开发外包给第三方（例如，供应商或研究人员）。第三方将其欺诈检测器提交给数据持有者，数据持有者在私人数据集上评估这些算法，然后公开传达结果。我们对该系统提出了一种现实的隐私攻击，允许对手仅根据评估结果对个人数据进行去匿名化。在美国国家标准与技术研究院（NIH）对面部识别算法隐私敏感基准的模拟中，我们的攻击在识别个人数据是否存在于私人数据集中方面实现了近乎完美的准确性，真阳性率为0.98，假阳性率为0.00。然后，我们研究如何在满足正式的差异隐私（DP）保证的同时对算法进行基准测试。我们根据经验评估了两类解决方案：子样本和聚合数据和DP合成图数据。我们通过大量实验证明，当前的方法在保证DP时无法提供实用性。我们的结果表明，DP引起的误差在扭曲图结构的偏差和添加随机噪音的方差之间权衡。当前的方法位于这种偏差方差权衡的不同点，但更复杂的方法往往需要添加高方差的噪音，从而削弱了效用。



## **35. Resilient State Recovery using Prior Measurement Support Information**

使用先前测量支持信息进行弹性状态恢复 math.OC

To be published in SIAM Journal on Control and Optimization

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22340v1) [paper-pdf](http://arxiv.org/pdf/2507.22340v1)

**Authors**: Yu Zheng, Olugbenga Moses Anubi, Warren E. Dixon

**Abstract**: Resilient state recovery of cyber-physical systems has attracted much research attention due to the unique challenges posed by the tight coupling between communication, computation, and the underlying physics of such systems. By modeling attacks as additive adversary signals to a sparse subset of measurements, this resilient recovery problem can be formulated as an error correction problem. To achieve exact state recovery, most existing results require less than $50\%$ of the measurement nodes to be compromised, which limits the resiliency of the estimators. In this paper, we show that observer resiliency can be further improved by incorporating data-driven prior information. We provide an analytical bridge between the precision of prior information and the resiliency of the estimator. By quantifying the relationship between the estimation error of the weighted $\ell_1$ observer and the precision of the support prior. This quantified relationship provides guidance for the estimator's weight design to achieve optimal resiliency. Several numerical simulations and an application case study are presented to validate the theoretical claims.

摘要: 由于通信、计算和此类系统的基础物理之间的紧密耦合所带来的独特挑战，网络物理系统的弹性状态恢复引起了广泛的研究关注。通过将攻击建模为对稀疏测量子集的添加对手信号，这个弹性恢复问题可以被表述为错误纠正问题。为了实现精确的状态恢复，大多数现有结果需要不到50%的测量节点受到损害，这限制了估计器的弹性。在本文中，我们表明可以通过纳入数据驱动的先验信息进一步提高观察者的弹性。我们在先验信息的精确性和估计器的弹性之间提供了分析桥梁。通过量化加权$\ell_1 $观察者的估计误差与支持先验精度之间的关系。这种量化关系为估计器的权重设计提供指导，以实现最佳弹性。文中给出了几个数值模拟和应用案例研究来验证理论主张。



## **36. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

22 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2411.08003v3) [paper-pdf](http://arxiv.org/pdf/2411.08003v3)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.

摘要: 在对抗环境中（例如网络攻击和虚假信息攻击）对大型语言模型（LLM）的输出进行归因会带来重大挑战，而且其重要性可能会越来越大。我们从理论和实证的角度来处理这个归因问题，借鉴形式语言理论（极限识别）和对不断扩大的LLM生态系统的数据驱动分析。通过将LLM的一组可能输出建模为形式语言，我们分析有限的文本样本是否可以唯一地确定原始模型。我们的结果表明，在模型之间能力重叠的温和假设下，某些类别的LLM从根本上无法仅从其输出中识别。我们描绘了理论可识别性的四种制度：（1）无限一类确定性（离散）LLM语言不可识别（Gold的经典结果来自1967年）;（2）无限类概率LLM也是不可识别的（通过确定性情况的扩展）;（3）有限类确定性LLM是可识别的（与Angluin的泄密标准一致）;以及（4）即使是有限类的概率LLM也可能是不可识别的（我们提供了一个新的反例来建立这个负结果）。作为对这些理论见解的补充，我们量化了近年来给定输出的合理模型起源（假设空间）数量的爆炸式增长。即使在保守的假设下--每个开源模型最多在一个新厕所上进行微调--不同候选模型的数量也大约每0.5年翻一番，并且允许多数据集微调组合可以产生翻倍的时间短至0.28年。这种组合增长，加上所有模型和潜在用户的暴力可能性归因的非凡计算成本，使得详尽的归因在实践中不可行。



## **37. Persistent Backdoor Attacks in Continual Learning**

持续学习中的持续后门攻击 cs.LG

19 pages, 20 figures, 6 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2409.13864v3) [paper-pdf](http://arxiv.org/pdf/2409.13864v3)

**Authors**: Zhen Guo, Abhinav Kumar, Reza Tourani

**Abstract**: Backdoor attacks pose a significant threat to neural networks, enabling adversaries to manipulate model outputs on specific inputs, often with devastating consequences, especially in critical applications. While backdoor attacks have been studied in various contexts, little attention has been given to their practicality and persistence in continual learning, particularly in understanding how the continual updates to model parameters, as new data distributions are learned and integrated, impact the effectiveness of these attacks over time. To address this gap, we introduce two persistent backdoor attacks-Blind Task Backdoor and Latent Task Backdoor-each leveraging minimal adversarial influence. Our blind task backdoor subtly alters the loss computation without direct control over the training process, while the latent task backdoor influences only a single task's training, with all other tasks trained benignly. We evaluate these attacks under various configurations, demonstrating their efficacy with static, dynamic, physical, and semantic triggers. Our results show that both attacks consistently achieve high success rates across different continual learning algorithms, while effectively evading state-of-the-art defenses, such as SentiNet and I-BAU.

摘要: 后门攻击对神经网络构成了重大威胁，使对手能够操纵特定输入的模型输出，通常会带来毁灭性的后果，特别是在关键应用中。虽然后门攻击已经在各种背景下进行了研究，但很少有人关注它们在持续学习中的实用性和持久性，特别是在了解随着新数据分布的学习和集成，模型参数的持续更新如何影响这些攻击的有效性方面。为了解决这一差距，我们引入了两种持续的后门攻击-盲任务后门和潜在任务后门-每一种都利用最小的对抗影响。我们的盲任务后门巧妙地改变了损失计算，而不直接控制训练过程，而潜在的任务后门只影响单个任务的训练，所有其他任务的训练都是良性的。我们在各种配置下评估了这些攻击，展示了它们在静态、动态、物理和语义触发下的功效。我们的结果表明，这两种攻击在不同的持续学习算法上始终实现了高成功率，同时有效地规避了SentiNet和I-BAU等最先进的防御。



## **38. Teach Me to Trick: Exploring Adversarial Transferability via Knowledge Distillation**

教我恶作剧：通过知识提炼探索对抗性可转移性 cs.LG

10 pages, 4 figures

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21992v1) [paper-pdf](http://arxiv.org/pdf/2507.21992v1)

**Authors**: Siddhartha Pradhan, Shikshya Shiwakoti, Neha Bathuri

**Abstract**: We investigate whether knowledge distillation (KD) from multiple heterogeneous teacher models can enhance the generation of transferable adversarial examples. A lightweight student model is trained using two KD strategies: curriculum-based switching and joint optimization, with ResNet50 and DenseNet-161 as teachers. The trained student is then used to generate adversarial examples using FG, FGS, and PGD attacks, which are evaluated against a black-box target model (GoogLeNet). Our results show that student models distilled from multiple teachers achieve attack success rates comparable to ensemble-based baselines, while reducing adversarial example generation time by up to a factor of six. An ablation study further reveals that lower temperature settings and the inclusion of hard-label supervision significantly enhance transferability. These findings suggest that KD can serve not only as a model compression technique but also as a powerful tool for improving the efficiency and effectiveness of black-box adversarial attacks.

摘要: 我们研究来自多个异类教师模型的知识提炼（KD）是否可以增强可转移对抗示例的生成。轻量级学生模型使用两种KD策略进行训练：基于课程的切换和联合优化，ResNet 50和DenseNet-161作为教师。然后，经过训练的学生使用FG、FSG和PVD攻击生成对抗性示例，并针对黑匣子目标模型（GoogLeNet）进行评估。我们的结果表明，从多名教师中提取的学生模型的攻击成功率与基于整体的基线相当，同时将对抗性示例生成时间减少六倍。一项消融研究进一步表明，较低的温度设置和硬标签监督的纳入显着增强了可转移性。这些发现表明，KD不仅可以作为一种模型压缩技术，还可以作为提高黑匣子对抗攻击效率和有效性的强大工具。



## **39. ZIUM: Zero-Shot Intent-Aware Adversarial Attack on Unlearned Models**

ZIUM：对未学习模型的零攻击意图感知对抗攻击 cs.CV

Accepted to ICCV2025

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21985v1) [paper-pdf](http://arxiv.org/pdf/2507.21985v1)

**Authors**: Hyun Jun Yook, Ga San Jhun, Jae Hyun Cho, Min Jeon, Donghyun Kim, Tae Hyung Kim, Youn Kyu Lee

**Abstract**: Machine unlearning (MU) removes specific data points or concepts from deep learning models to enhance privacy and prevent sensitive content generation. Adversarial prompts can exploit unlearned models to generate content containing removed concepts, posing a significant security risk. However, existing adversarial attack methods still face challenges in generating content that aligns with an attacker's intent while incurring high computational costs to identify successful prompts. To address these challenges, we propose ZIUM, a Zero-shot Intent-aware adversarial attack on Unlearned Models, which enables the flexible customization of target attack images to reflect an attacker's intent. Additionally, ZIUM supports zero-shot adversarial attacks without requiring further optimization for previously attacked unlearned concepts. The evaluation across various MU scenarios demonstrated ZIUM's effectiveness in successfully customizing content based on user-intent prompts while achieving a superior attack success rate compared to existing methods. Moreover, its zero-shot adversarial attack significantly reduces the attack time for previously attacked unlearned concepts.

摘要: 机器去学习（MU）从深度学习模型中删除特定数据点或概念，以增强隐私并防止敏感内容生成。对抗性提示可以利用未学习的模型来生成包含已删除概念的内容，从而构成重大的安全风险。然而，现有的对抗性攻击方法在生成符合攻击者意图的内容方面仍然面临挑战，同时识别成功提示需要付出高昂的计算成本。为了应对这些挑战，我们提出了ZIUM，这是一种对Unleared Models的零射击意图感知对抗攻击，它能够灵活定制目标攻击图像以反映攻击者的意图。此外，ZIUM支持零射击对抗攻击，而不需要进一步优化先前攻击的未学习概念。对各种MU场景的评估表明，ZIUM在根据用户意图提示成功定制内容方面的有效性，同时与现有方法相比，其攻击成功率更高。此外，它的零射击对抗攻击大大减少了以前攻击的未学习概念的攻击时间。



## **40. Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is**

任何人都可以越狱：针对LLM和T2 I的预算攻击 cs.CV

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21820v1) [paper-pdf](http://arxiv.org/pdf/2507.21820v1)

**Authors**: Ahmed B Mustafa, Zihan Ye, Yang Lu, Michael P Pound, Shreyank N Gowda

**Abstract**: Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings.

摘要: 尽管在对齐和内容审核方面取得了重大进步，但大型语言模型（LLM）和文本到图像（T2 I）系统仍然容易受到基于预算的攻击（即越狱）。与需要专业知识的传统对抗示例不同，今天的许多越狱都是由日常用户精心设计的，只需措辞巧妙的提示即可。本文对非专家如何通过多回合叙事升级、词汇伪装、隐含链接、虚构模仿和微妙的语义编辑等技术可靠地规避安全机制进行了系统式的调查。我们基于流行API的实证案例研究，提出了跨越文本输出和T2 I模型的预算级越狱策略的统一分类。我们的分析表明，审核管道的每个阶段，从输入过滤到输出验证，都可以通过可访问的策略绕过。最后，我们强调了对上下文感知防御的迫切需要，以反映这些越狱可以在现实世界环境中轻松复制的情况。



## **41. Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal**

没有对抗防御的对抗防御：通过实例级主成分去除增强语言模型稳健性 cs.CL

This paper was accepted with an A-decision to Transactions of the  Association for Computational Linguistics. This version is the  pre-publication version prior to MIT Press production

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21750v1) [paper-pdf](http://arxiv.org/pdf/2507.21750v1)

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.

摘要: 预训练的语言模型（PLM）推动了自然语言处理的重大进展，但仍然容易受到对抗攻击，引发了对其在现实世界应用程序中稳健性的担忧。之前的研究试图通过隐式或显式地在训练过程中引入对抗性扰动来减轻对抗性攻击的影响。虽然这两种策略都增强了稳健性，但它们通常会产生很高的计算成本。在这项工作中，我们提出了一个简单而有效的附加模块，该模块通过删除实例级主成分来增强PLM的对抗鲁棒性，而不依赖于传统的对抗防御或干扰原始训练数据。我们的方法将嵌入空间转换为逼近高斯属性，从而降低其对对抗性扰动的敏感性，同时保留语义关系。这种转换以一种最小化对抗性噪音对决策边界的影响的方式对齐嵌入分布，增强稳健性，而无需对抗性示例或昂贵的训练时间扩展。对八个基准数据集的评估表明，我们的方法提高了对抗稳健性，同时保持了与基线相当的攻击前准确性，实现了稳健性和概括性之间的平衡。



## **42. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

通过潜在对抗训练防御不可预见的失败模式 cs.CR

See also followup work at arXiv:2407.15549

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2403.05030v6) [paper-pdf](http://arxiv.org/pdf/2403.05030v6)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without leveraging knowledge of what they are or using inputs that elicit them. LAT makes use of the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. Here, we use it to defend against failure modes without examples that elicit them. Specifically, we use LAT to remove backdoors and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.

摘要: 尽管开发人员进行了广泛的诊断和调试，人工智能系统有时会表现出有害的非预期行为。找到和修复这些问题具有挑战性，因为攻击面如此之大--无法彻底搜索可能引发有害行为的输入。红色团队和对抗训练（AT）通常用于提高稳健性，然而，从经验上看，它们很难修复与训练期间使用的攻击不同的失败模式。在这项工作中，我们利用潜在对抗训练（LAT）来防御漏洞，而无需利用有关漏洞的知识或使用引发漏洞的输入。LAT利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示。在这里，我们使用它来防御没有引发失败模式的例子的失败模式。具体来说，我们使用LAT来删除后门并抵御持续的对抗性攻击。我们在图像分类、文本分类和文本生成任务中表明，相对于AT，LAT通常会提高对新型攻击的鲁棒性和干净数据的性能。这表明LAT可以成为一种有前途的工具，用于防御开发人员未明确识别的故障模式。



## **43. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

隐性对抗培训提高了法学硕士对持续有害行为的稳健性 cs.LG

Code at https://github.com/aengusl/latent-adversarial-training.  Models at https://huggingface.co/LLM-LAT

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2407.15549v3) [paper-pdf](http://arxiv.org/pdf/2407.15549v3)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型（LLM）通常会以不受欢迎的方式运行，而这些方式明确进行了微调。例如，LLM红色团队文献产生了各种“越狱”技术，从经过微调的无害模型中引出有害文本。最近关于红色团队、模型编辑和可解释性的工作表明，这一挑战源于（对抗性）微调如何在很大程度上用于抑制而不是消除LLM中不受欢迎的功能。之前的工作引入了潜在对抗训练（LAT），作为提高对广泛类型失败的稳健性的一种方法。这些先前的作品考虑了无针对性的潜在空间攻击，其中对手扰乱潜在激活，以最大限度地增加理想行为示例的损失。无目标LAT可以提供通用类型的稳健性，但不会利用有关特定故障模式的信息。在这里，我们尝试了有针对性的LAT，其中对手试图最大限度地减少特定竞争任务的损失。我们发现它可以增强各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的稳健性，以减少数量级的计算来超越强大的R2 D2基线。其次，我们使用它来在不知道触发器的情况下更有效地删除后门。最后，我们使用它以一种对重新学习更稳健的方式更有效地忘记特定不需要任务的知识。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **44. PRISM: Programmatic Reasoning with Image Sequence Manipulation for LVLM Jailbreaking**

PRism：用于LVLM越狱的具有图像序列操作的程序推理 cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21540v1) [paper-pdf](http://arxiv.org/pdf/2507.21540v1)

**Authors**: Quanchen Zou, Zonghao Ying, Moyang Chen, Wenzhuo Xu, Yisong Xiao, Yakai Li, Deyue Zhang, Dongdong Yang, Zhao Liu, Xiangzheng Zhang

**Abstract**: The increasing sophistication of large vision-language models (LVLMs) has been accompanied by advances in safety alignment mechanisms designed to prevent harmful content generation. However, these defenses remain vulnerable to sophisticated adversarial attacks. Existing jailbreak methods typically rely on direct and semantically explicit prompts, overlooking subtle vulnerabilities in how LVLMs compose information over multiple reasoning steps. In this paper, we propose a novel and effective jailbreak framework inspired by Return-Oriented Programming (ROP) techniques from software security. Our approach decomposes a harmful instruction into a sequence of individually benign visual gadgets. A carefully engineered textual prompt directs the sequence of inputs, prompting the model to integrate the benign visual gadgets through its reasoning process to produce a coherent and harmful output. This makes the malicious intent emergent and difficult to detect from any single component. We validate our method through extensive experiments on established benchmarks including SafeBench and MM-SafetyBench, targeting popular LVLMs. Results show that our approach consistently and substantially outperforms existing baselines on state-of-the-art models, achieving near-perfect attack success rates (over 0.90 on SafeBench) and improving ASR by up to 0.39. Our findings reveal a critical and underexplored vulnerability that exploits the compositional reasoning abilities of LVLMs, highlighting the urgent need for defenses that secure the entire reasoning process.

摘要: 随着大型视觉语言模型（LVLM）的日益复杂，旨在防止有害内容生成的安全对齐机制也取得了进步。然而，这些防御系统仍然容易受到复杂的对抗攻击。现有的越狱方法通常依赖于直接且语义明确的提示，忽略了LVLM如何通过多个推理步骤组成信息的微妙漏洞。本文受到软件安全领域的面向返回编程（opp）技术的启发，提出了一种新颖且有效的越狱框架。我们的方法将有害的指令分解为一系列单独良性的视觉小工具。精心设计的文本提示引导输入序列，促使模型通过其推理过程集成良性视觉小工具，以产生连贯且有害的输出。这使得恶意意图变得紧急，并且难以从任何单个组件中检测到。我们通过对SafeBench和MM-SafetyBench等既定基准进行广泛实验来验证我们的方法，目标是流行的LVLM。结果表明，我们的方法始终且大幅优于最先进模型上的现有基线，实现了近乎完美的攻击成功率（SafeBench上超过0.90），并将ASB提高高达0.39。我们的研究结果揭示了一个关键且未充分探索的漏洞，该漏洞利用了LVLM的合成推理能力，凸显了对保护整个推理过程的防御措施的迫切需求。



## **45. Can We End the Cat-and-Mouse Game? Simulating Self-Evolving Phishing Attacks with LLMs and Genetic Algorithms**

我们能结束猫鼠游戏吗？使用LLM和遗传算法模拟自我进化的网络钓鱼攻击 cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21538v1) [paper-pdf](http://arxiv.org/pdf/2507.21538v1)

**Authors**: Seiji Sato, Tetsushi Ohki, Masakatsu Nishigaki

**Abstract**: Anticipating emerging attack methodologies is crucial for proactive cybersecurity. Recent advances in Large Language Models (LLMs) have enabled the automated generation of phishing messages and accelerated research into potential attack techniques. However, predicting future threats remains challenging due to reliance on existing training data. To address this limitation, we propose a novel framework that integrates LLM-based phishing attack simulations with a genetic algorithm in a psychological context, enabling phishing strategies to evolve dynamically through adversarial interactions with simulated victims. Through simulations using Llama 3.1, we demonstrate that (1) self-evolving phishing strategies employ increasingly sophisticated psychological manipulation techniques, surpassing naive LLM-generated attacks, (2) variations in a victim's prior knowledge significantly influence the evolution of attack strategies, and (3) adversarial interactions between evolving attacks and adaptive defenses create a cat-and-mouse dynamic, revealing an inherent asymmetry in cybersecurity -- attackers continuously refine their methods, whereas defenders struggle to comprehensively counter all evolving threats. Our approach provides a scalable, cost-effective method for analyzing the evolution of phishing strategies and defenses, offering insights into future social engineering threats and underscoring the necessity of proactive cybersecurity measures.

摘要: 预测新出现的攻击方法对于主动网络安全至关重要。大型语言模型（LLM）的最新进展使网络钓鱼消息的自动生成成为可能，并加速了对潜在攻击技术的研究。然而，由于依赖于现有的训练数据，预测未来的威胁仍然具有挑战性。为了解决这一限制，我们提出了一种新的框架，集成了基于LLM的网络钓鱼攻击模拟与遗传算法在心理背景下，使网络钓鱼策略通过与模拟受害者的对抗性交互动态演变。通过使用Llama 3.1的模拟，我们证明了（1）自我进化的网络钓鱼策略采用了越来越复杂的心理操纵技术，超越了天真的LLM生成的攻击，（2）受害者先验知识的变化显着影响攻击策略的演变，（3）不断进化的攻击和适应性防御之间的对抗相互作用创造了猫鼠动态，暴露了网络安全固有的不对称性--攻击者不断完善他们的方法，而防御者则难以全面应对所有不断变化的威胁。我们的方法提供了一种可扩展、具有成本效益的方法来分析网络钓鱼策略和防御的演变，提供了对未来社会工程威胁的见解，并强调了主动网络安全措施的必要性。



## **46. NCCR: to Evaluate the Robustness of Neural Networks and Adversarial Examples**

NCCR：评估神经网络和对抗示例的鲁棒性 cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21483v1) [paper-pdf](http://arxiv.org/pdf/2507.21483v1)

**Authors**: Pu Shi

**Abstract**: Neural networks have received a lot of attention recently, and related security issues have come with it. Many studies have shown that neural networks are vulnerable to adversarial examples that have been artificially perturbed with modification, which is too small to be distinguishable by human perception. Different attacks and defenses have been proposed to solve these problems, but there is little research on evaluating the robustness of neural networks and their inputs. In this work, we propose a metric called the neuron cover change rate (NCCR) to measure the ability of deep learning models to resist attacks and the stability of adversarial examples. NCCR monitors alterations in the output of specifically chosen neurons when the input is perturbed, and networks with a smaller degree of variation are considered to be more robust. The results of the experiment on image recognition and the speaker recognition model show that our metrics can provide a good assessment of the robustness of neural networks or their inputs. It can also be used to detect whether an input is adversarial or not, as adversarial examples are always less robust.

摘要: 神经网络最近受到了广泛关注，相关的安全问题也随之而来。许多研究表明，神经网络容易受到经过修改人为干扰的对抗性示例的影响，这些示例太小，无法通过人类感知来区分。人们提出了不同的攻击和防御来解决这些问题，但关于评估神经网络及其输入的稳健性的研究很少。在这项工作中，我们提出了一种名为神经元覆盖变化率（NCCR）的指标来衡量深度学习模型抵抗攻击的能力和对抗性示例的稳定性。当输入受到干扰时，NCCR监控特定选择的神经元输出的变化，并且变化程度较小的网络被认为更稳健。图像识别和说话人识别模型的实验结果表明，我们的指标可以很好地评估神经网络或其输入的鲁棒性。它还可以用于检测输入是否具有对抗性，因为对抗性示例总是不太稳健。



## **47. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

PAR-AdvGAN：通过渐进式自回归AdvGAN提高对抗攻击能力 cs.LG

Best paper award of ECML-PKDD 2025

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2502.12207v2) [paper-pdf](http://arxiv.org/pdf/2502.12207v2)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://github.com/LMBTough/PAR

摘要: 深度神经网络在各个领域都表现出了卓越的性能。然而，它们容易受到对抗性例子的影响，这可能导致错误的预测。生成对抗网络（GAN）可以利用生成器和鉴别器模型快速生成高质量的对抗示例。由于两个模块都以竞争和同步的方式进行训练，因此与传统方法相比，基于GAN的算法（如AdvGAN）可以生成具有更好可移植性的对抗性示例。然而，扰动的产生通常仅限于单次迭代，从而阻止这些示例充分利用方法的潜力。为了解决这个问题，我们引入了一种名为渐进式自动回归AdvGAN（PAR-AdvGAN）的新颖方法。它在渐进生成网络中集成了自回归迭代机制，以制作具有增强攻击能力的对抗性示例。我们通过大规模实验彻底评估了我们的PAR-AdvGAN方法，证明了其优于各种最先进的黑匣子对抗攻击以及原始的AdvGAN的性能。此外，PAR-AdvGAN显着加速了对抗性示例的生成，即Inception-v3模型上的速度高达每秒335.5帧，优于基于梯度的可转移攻击算法。我们的代码可访问：https://github.com/LMBTough/PAR



## **48. Cascading and Proxy Membership Inference Attacks**

级联和代理成员推断攻击 cs.CR

Our code is available at: https://github.com/zealscott/MIA

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21412v1) [paper-pdf](http://arxiv.org/pdf/2507.21412v1)

**Authors**: Yuntao Du, Jiacheng Li, Yuetian Chen, Kaiyuan Zhang, Zhizhen Yuan, Hanshen Xiao, Bruno Ribeiro, Ninghui Li

**Abstract**: A Membership Inference Attack (MIA) assesses how much a trained machine learning model reveals about its training data by determining whether specific query instances were included in the dataset. We classify existing MIAs into adaptive or non-adaptive, depending on whether the adversary is allowed to train shadow models on membership queries. In the adaptive setting, where the adversary can train shadow models after accessing query instances, we highlight the importance of exploiting membership dependencies between instances and propose an attack-agnostic framework called Cascading Membership Inference Attack (CMIA), which incorporates membership dependencies via conditional shadow training to boost membership inference performance.   In the non-adaptive setting, where the adversary is restricted to training shadow models before obtaining membership queries, we introduce Proxy Membership Inference Attack (PMIA). PMIA employs a proxy selection strategy that identifies samples with similar behaviors to the query instance and uses their behaviors in shadow models to perform a membership posterior odds test for membership inference. We provide theoretical analyses for both attacks, and extensive experimental results demonstrate that CMIA and PMIA substantially outperform existing MIAs in both settings, particularly in the low false-positive regime, which is crucial for evaluating privacy risks.

摘要: 成员资格推理攻击（MIA）通过确定数据集中是否包括特定的查询实例来评估经过训练的机器学习模型对其训练数据的揭示程度。我们将现有的MIA分为自适应或非自适应，具体取决于是否允许对手在成员资格查询上训练影子模型。在自适应环境中，对手可以在访问查询实例后训练影子模型，我们强调了利用实例之间成员依赖关系的重要性，并提出了一种名为级联成员推断攻击（CMIA）的攻击不可知框架，该框架通过条件影子训练合并成员依赖关系，以提高成员推断性能。   在非自适应环境中，对手仅限于在获得成员资格查询之前训练影子模型，我们引入代理成员资格推断攻击（PMIA）。PMIA采用代理选择策略，该策略识别与查询实例具有相似行为的样本，并使用其在影子模型中的行为来执行成员资格后验赔率测试以进行成员资格推断。我们对这两种攻击提供了理论分析，大量的实验结果表明，CMIA和PMIA在这两种环境下的表现都大大优于现有的MIA，特别是在低假阳性机制下，这对于评估隐私风险至关重要。



## **49. FedStrategist: A Meta-Learning Framework for Adaptive and Robust Aggregation in Federated Learning**

FedStrategist：一个用于联邦学习中自适应和鲁棒聚合的元学习框架 cs.LG

24 pages, 8 figures. This work is intended for a journal submission

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.14322v2) [paper-pdf](http://arxiv.org/pdf/2507.14322v2)

**Authors**: Md Rafid Haque, Abu Raihan Mostofa Kamal, Md. Azam Hossain

**Abstract**: Federated Learning (FL) offers a paradigm for privacy-preserving collaborative AI, but its decentralized nature creates significant vulnerabilities to model poisoning attacks. While numerous static defenses exist, their effectiveness is highly context-dependent, often failing against adaptive adversaries or in heterogeneous data environments. This paper introduces FedStrategist, a novel meta-learning framework that reframes robust aggregation as a real-time, cost-aware control problem. We design a lightweight contextual bandit agent that dynamically selects the optimal aggregation rule from an arsenal of defenses based on real-time diagnostic metrics. Through comprehensive experiments, we demonstrate that no single static rule is universally optimal. We show that our adaptive agent successfully learns superior policies across diverse scenarios, including a ``Krum-favorable" environment and against a sophisticated "stealth" adversary designed to neutralize specific diagnostic signals. Critically, we analyze the paradoxical scenario where a non-robust baseline achieves high but compromised accuracy, and demonstrate that our agent learns a conservative policy to prioritize model integrity. Furthermore, we prove the agent's policy is controllable via a single "risk tolerance" parameter, allowing practitioners to explicitly manage the trade-off between performance and security. Our work provides a new, practical, and analyzable approach to creating resilient and intelligent decentralized AI systems.

摘要: 联邦学习（FL）为保护隐私的协作人工智能提供了一个范式，但其去中心化性质给建模中毒攻击带来了显着的漏洞。虽然存在许多静态防御，但它们的有效性高度依赖于上下文，通常无法对抗自适应对手或在异类数据环境中。本文介绍了FedStrategist，这是一种新型的元学习框架，它将稳健聚合重新定义为实时、成本感知的控制问题。我们设计了一个轻量级的上下文强盗代理，它基于实时诊断指标从防御库中动态选择最佳聚合规则。通过全面的实验，我们证明没有单一的静态规则是普遍最优的。我们表明，我们的适应性代理能够在不同的场景中成功学习更好的策略，包括“克鲁姆有利”的环境以及针对旨在中和特定诊断信号的复杂“隐形”对手。至关重要的是，我们分析了自相矛盾的场景，即非稳健基线实现了高但受到损害的准确性，并证明我们的代理学习了保守的策略来优先考虑模型完整性。此外，我们证明了代理的策略是可以通过单个“风险容忍度”参数来控制的，允许从业者显式地管理性能和安全性之间的权衡。我们的工作提供了一种新的、实用的、可分析的方法来创建弹性和智能的去中心化人工智能系统。



## **50. Radio Adversarial Attacks on EMG-based Gesture Recognition Networks**

基于EMG的手势识别网络的无线对抗攻击 cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21387v1) [paper-pdf](http://arxiv.org/pdf/2507.21387v1)

**Authors**: Hongyi Xie

**Abstract**: Surface electromyography (EMG) enables non-invasive human-computer interaction in rehabilitation, prosthetics, and virtual reality. While deep learning models achieve over 97% classification accuracy, their vulnerability to adversarial attacks remains largely unexplored in the physical domain. We present ERa Attack, the first radio frequency (RF) adversarial method targeting EMG devices through intentional electromagnetic interference (IEMI). Using low-power software-defined radio transmitters, attackers inject optimized RF perturbations to mislead downstream models. Our approach bridges digital and physical domains: we generate adversarial perturbations using Projected Gradient Descent, extract 50-150 Hz components via inverse STFT, and employ synchronization-free strategies (constant spectrum noise or narrowband modulation). Perturbations, constrained to 1-10% of signal amplitude, are amplitude-modulated onto 433 MHz carriers. Experiments on the Myo Dataset (7 gestures, 350 samples) demonstrate significant impact: at 1 meter and 0 dBm transmission power, classification accuracy drops from 97.8% to 58.3%, with 41.7% misclassification rate and 25.6% targeted attack success rate. Attack effectiveness decreases exponentially with distance, recovering to 85% accuracy at 3 meters. Increasing power to 10 dBm reduces accuracy by an additional 15% at 1 meter. This work pioneers RF-based adversarial attacks on EMG recognition systems, revealing critical vulnerabilities in safety-critical applications. We quantify attack effectiveness across different perturbation modes and distances, and propose defenses including hardware shielding, spectrum monitoring, and adversarial training. Our findings inform the design of robust EMG systems against electromagnetic threats.

摘要: 表面肌电信号（EMG）实现康复、假肢和虚拟现实中的非侵入性人机交互。虽然深度学习模型的分类准确率超过97%，但它们对对抗攻击的脆弱性在物理领域基本上尚未被探索。我们介绍了ERa Attack，这是第一种通过故意电磁干扰（IEMI）针对EMG设备的射频（RF）对抗方法。使用低功耗软件定义的无线电发射机，攻击者注入优化的RF扰动来误导下游模型。我们的方法架起数字和物理领域的桥梁：我们使用投影梯度下降来生成对抗性扰动，通过逆STFT提取50-150 Hz分量，并采用无同步策略（恒定频谱噪音或窄频调制）。限制在信号幅度的1-10%的扰动被幅度调制到433 MHz载体上。Myo数据集（7个手势，350个样本）的实验显示了显着的影响：在1米和0 dBm传输功率下，分类准确率从97.8%下降到58.3%，误分类率为41.7%，目标攻击成功率为25.6%。攻击有效性随着距离的增加呈指数级下降，在3米处恢复到85%的准确率。将功率增加到10分贝会使1米处的准确性额外降低15%。这项工作开创了对EMG识别系统的基于RF的对抗攻击，揭示了安全关键应用程序中的关键漏洞。我们量化了不同扰动模式和距离的攻击有效性，并提出了包括硬件屏蔽、频谱监控和对抗训练在内的防御措施。我们的研究结果为针对电磁威胁的稳健EMG系统的设计提供了信息。



