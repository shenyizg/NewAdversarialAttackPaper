# Latest Adversarial Attack Papers
**update at 2025-04-23 10:05:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Observations in Weather Forecasting**

天气预报中的对抗性观测 cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15942v1) [paper-pdf](http://arxiv.org/pdf/2504.15942v1)

**Authors**: Erik Imgrund, Thorsten Eisenhofer, Konrad Rieck

**Abstract**: AI-based systems, such as Google's GenCast, have recently redefined the state of the art in weather forecasting, offering more accurate and timely predictions of both everyday weather and extreme events. While these systems are on the verge of replacing traditional meteorological methods, they also introduce new vulnerabilities into the forecasting process. In this paper, we investigate this threat and present a novel attack on autoregressive diffusion models, such as those used in GenCast, capable of manipulating weather forecasts and fabricating extreme events, including hurricanes, heat waves, and intense rainfall. The attack introduces subtle perturbations into weather observations that are statistically indistinguishable from natural noise and change less than 0.1% of the measurements - comparable to tampering with data from a single meteorological satellite. As modern forecasting integrates data from nearly a hundred satellites and many other sources operated by different countries, our findings highlight a critical security risk with the potential to cause large-scale disruptions and undermine public trust in weather prediction.

摘要: 谷歌的GenCast等基于人工智能的系统最近重新定义了天气预报的最新水平，为日常天气和极端事件提供了更准确、更及时的预测。虽然这些系统即将取代传统的气象方法，但它们也给预报过程带来了新的漏洞。在本文中，我们调查了这一威胁，并对自回归扩散模型（例如GenCast中使用的模型）提出了一种新颖的攻击，这些模型能够操纵天气预报并编造极端事件，包括飓风、热浪和强降雨。该攻击在天气观测中引入了微妙的扰动，这些扰动在统计上与自然噪音难以区分，并且测量结果的变化不到0.1%--与篡改单个气象卫星的数据相当。由于现代预报整合了来自不同国家运营的近一百颗卫星和许多其他来源的数据，我们的研究结果凸显了一个关键的安全风险，有可能造成大规模干扰并破坏公众对天气预测的信任。



## **2. Human-Imperceptible Physical Adversarial Attack for NIR Face Recognition Models**

近红外人脸识别模型的人类不可感知的物理对抗攻击 cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15823v1) [paper-pdf](http://arxiv.org/pdf/2504.15823v1)

**Authors**: Songyan Xie, Jinghang Wen, Encheng Su, Qiucheng Yu

**Abstract**: Near-infrared (NIR) face recognition systems, which can operate effectively in low-light conditions or in the presence of makeup, exhibit vulnerabilities when subjected to physical adversarial attacks. To further demonstrate the potential risks in real-world applications, we design a novel, stealthy, and practical adversarial patch to attack NIR face recognition systems in a black-box setting. We achieved this by utilizing human-imperceptible infrared-absorbing ink to generate multiple patches with digitally optimized shapes and positions for infrared images. To address the optimization mismatch between digital and real-world NIR imaging, we develop a light reflection model for human skin to minimize pixel-level discrepancies by simulating NIR light reflection.   Compared to state-of-the-art (SOTA) physical attacks on NIR face recognition systems, the experimental results show that our method improves the attack success rate in both digital and physical domains, particularly maintaining effectiveness across various face postures. Notably, the proposed approach outperforms SOTA methods, achieving an average attack success rate of 82.46% in the physical domain across different models, compared to 64.18% for existing methods. The artifact is available at https://anonymous.4open.science/r/Human-imperceptible-adversarial-patch-0703/.

摘要: 近红外（NIR）人脸识别系统可以在弱光条件下或化妆时有效工作，但在受到物理对抗攻击时会表现出脆弱性。为了进一步证明现实世界应用程序中的潜在风险，我们设计了一种新颖、隐蔽且实用的对抗补丁来攻击黑匣子环境中的近红外人脸识别系统。我们通过利用人类难以感知的红外吸收墨水来生成具有数字优化的红外图像形状和位置的多个补丁来实现这一目标。为了解决数字和现实世界的近红外成像之间的优化不匹配问题，我们开发了一种用于人类皮肤的光反射模型，通过模拟近红外光反射来最大限度地减少像素级差异。   与对近红外人脸识别系统的最新技术（SOTA）物理攻击相比，实验结果表明，我们的方法提高了数字和物理领域的攻击成功率，特别是在各种面部姿势下保持有效性。值得注意的是，提出的方法优于SOTA方法，在不同模型的物理域中实现了82.46%的平均攻击成功率，而现有方法的平均攻击成功率为64.18%。该产品可在https://anonymous.4open.science/r/Human-imperceptible-adversarial-patch-0703/上找到。



## **3. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

下一代物联网的图形神经网络：最近的进展和开放的挑战 cs.IT

28 pages, 15 figures, and 6 tables. Submitted for publication

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2412.20634v2) [paper-pdf](http://arxiv.org/pdf/2412.20634v2)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a critical tool for optimizing and managing the complexities of the Internet of Things (IoT) in next-generation networks. This survey presents a comprehensive exploration of how GNNs may be harnessed in 6G IoT environments, focusing on key challenges and opportunities through a series of open questions. We commence with an exploration of GNN paradigms and the roles of node, edge, and graph-level tasks in solving wireless networking problems and highlight GNNs' ability to overcome the limitations of traditional optimization methods. This guidance enhances problem-solving efficiency across various next-generation (NG) IoT scenarios. Next, we provide a detailed discussion of the application of GNN in advanced NG enabling technologies, including massive MIMO, reconfigurable intelligent surfaces, satellites, THz, mobile edge computing (MEC), and ultra-reliable low latency communication (URLLC). We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms to secure GNN-based NG-IoT networks. Next, we examine how GNNs can be integrated with future technologies like integrated sensing and communication (ISAC), satellite-air-ground-sea integrated networks (SAGSIN), and quantum computing. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security within NG-IoT systems, paving the way for future advances. Finally, we propose a set of design guidelines to facilitate the development of efficient, scalable, and secure GNN models tailored for NG IoT applications.

摘要: 图形神经网络（GNN）已成为优化和管理下一代网络中物联网（IoT）复杂性的重要工具。这项调查全面探索了如何在6 G物联网环境中利用GNN，并通过一系列开放性问题重点关注关键挑战和机遇。我们首先探索GNN范式以及节点、边缘和图形级任务在解决无线网络问题中的作用，并强调GNN克服传统优化方法局限性的能力。该指南提高了各种下一代（NG）物联网场景中的问题解决效率。接下来，我们详细讨论GNN在先进NG使能技术中的应用，包括大规模MMO、可重配置智能表面、卫星、太赫兹、移动边缘计算（MEC）和超可靠低延迟通信（URLLC）。然后，我们深入研究了对抗性攻击带来的挑战，深入了解保护基于GNN的NG-物联网网络的防御机制。接下来，我们研究GNN如何与集成传感和通信（ISAC）、卫星-空-地-海综合网络（SAGSIN）和量子计算等未来技术集成。我们的研究结果强调了GNN在提高NG-物联网系统内的效率、可扩展性和安全性方面的变革潜力，为未来的发展铺平了道路。最后，我们提出了一套设计准则，以促进开发为NG物联网应用量身定制的高效、可扩展且安全的GNN模型。



## **4. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多模式大型语言模型（MLLM）在各种多模式任务中展示了令人印象深刻的性能。另一方面，额外图像形态的集成可能会允许恶意用户在图像中注入有害内容以进行越狱。与基于文本的LLM不同，对手需要选择离散令牌来使用特定算法隐藏其恶意意图，图像信号的连续性为对手提供了注入有害意图的直接机会。在这项工作中，我们提出了$\textBF{BaThe}$（$\textBF{BA}$ckdoor $\textBF{T}$rigger S$\textBF{h}$i$\textBF{e}$ld），这是一种简单而有效的越狱防御机制。我们的工作受到最近对生成性语言模型中越狱后门攻击和虚拟提示后门攻击的研究的启发。越狱后门攻击使用有害指令与手工制作的字符串相结合作为触发器，使后门模型生成禁止的响应。我们假设有害指令可以充当触发器，如果我们将拒绝响应设置为触发响应，那么后门模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一目标，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为“wedge”。我们的全面实验表明，BaThe有效地缓解了各种类型的越狱攻击，并且能够抵御不可见的攻击，对MLLM的性能影响最小。



## **5. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

Red Team Diffuser：通过强化学习暴露视觉语言模型中的有毒连续漏洞 cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.

摘要: 大型视觉语言模型（VLM）的不断增加的部署暴露了其对齐机制中的关键安全漏洞。虽然现有的越狱研究主要关注VLM对有害指令的敏感性，但我们揭示了一个基本但被忽视的弱点：有毒文本延续，当提示有害文本前置与语义对抗图像配对时，VLM会产生剧毒的完成。为了系统性地研究这种威胁，我们提出了Red Team Distuser（RTI），这是第一个红色团队扩散模型，通过强化学习协调对抗图像生成和有毒延续。我们的关键创新包括动态跨模式攻击和隐身优化。对于LLM安全基准中的有毒文本前置，我们进行贪婪搜索以识别最大限度地引发有毒完成的最佳图像提示。发现的图像提示然后驱动基于RL的扩散模型微调，产生语义对齐的对抗图像，从而提高毒性率。潜行感知优化引入了联合对抗奖励，平衡毒性最大化（通过Dealfy分类器）和潜行性（通过BERTScore），规避了传统的基于噪音的对抗模式。实验结果证明了RTI的有效性，在原始攻击集中，LLaVA输出的毒性率比纯文本基线增加了10.69%，在未见集上增加了8.91%，证明了概括能力。此外，RTI具有较强的跨模型转移性，使Gemini的毒性率提高了5.1%，对LLaMA的毒性率提高了26.83%。我们的研究结果暴露了当前VLM对齐中的两个关键缺陷：（1）未能防止有害前置的有毒延续，以及（2）忽视了跨模式攻击载体。这些结果需要在安全性评估中向多模式红色团队转变。



## **6. TrojanDam: Detection-Free Backdoor Defense in Federated Learning through Proactive Model Robustification utilizing OOD Data**

TrojanDam：通过利用OOD数据的主动模型Robusification在联邦学习中实现无检测后门防御 cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15674v1) [paper-pdf](http://arxiv.org/pdf/2504.15674v1)

**Authors**: Yanbo Dai, Songze Li, Zihan Gan, Xueluan Gong

**Abstract**: Federated learning (FL) systems allow decentralized data-owning clients to jointly train a global model through uploading their locally trained updates to a centralized server. The property of decentralization enables adversaries to craft carefully designed backdoor updates to make the global model misclassify only when encountering adversary-chosen triggers. Existing defense mechanisms mainly rely on post-training detection after receiving updates. These methods either fail to identify updates which are deliberately fabricated statistically close to benign ones, or show inconsistent performance in different FL training stages. The effect of unfiltered backdoor updates will accumulate in the global model, and eventually become functional. Given the difficulty of ruling out every backdoor update, we propose a backdoor defense paradigm, which focuses on proactive robustification on the global model against potential backdoor attacks. We first reveal that the successful launching of backdoor attacks in FL stems from the lack of conflict between malicious and benign updates on redundant neurons of ML models. We proceed to prove the feasibility of activating redundant neurons utilizing out-of-distribution (OOD) samples in centralized settings, and migrating to FL settings to propose a novel backdoor defense mechanism, TrojanDam. The proposed mechanism has the FL server continuously inject fresh OOD mappings into the global model to activate redundant neurons, canceling the effect of backdoor updates during aggregation. We conduct systematic and extensive experiments to illustrate the superior performance of TrojanDam, over several SOTA backdoor defense methods across a wide range of FL settings.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **7. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

通过跨模式提示注射操纵多模式代理 cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.14348v2) [paper-pdf](http://arxiv.org/pdf/2504.14348v2)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.

摘要: 多模式大型语言模型的出现通过将语言和视觉模式与外部数据源集成来重新定义了代理范式，使代理能够更好地解释人类指令并执行日益复杂的任务。然而，在这项工作中，我们发现了多模式代理中一个以前被忽视的关键安全漏洞：跨模式提示注入攻击。为了利用这个漏洞，我们提出了CrossInib，这是一种新型攻击框架，其中攻击者在多种模式中嵌入对抗性扰动，以与目标恶意内容保持一致，允许外部指令劫持代理的决策过程并执行未经授权的任务。我们的方法由两个关键部分组成。首先，我们引入了视觉潜在对齐，基于文本到图像生成模型，优化视觉嵌入空间中恶意指令的对抗特征，确保对抗图像巧妙地编码恶意任务执行的线索。随后，我们提出了文本指导增强，其中利用大型语言模型通过对抗性Meta提示来推断黑匣子防御系统提示，并生成恶意文本命令，该命令引导代理的输出更好地遵守攻击者的请求。大量实验表明，我们的方法优于现有的注入攻击，在不同任务中的攻击成功率至少增加了+26.4%。此外，我们还验证了攻击在现实世界的多模式自治代理中的有效性，强调了其对安全关键应用程序的潜在影响。



## **8. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

Gungnir：利用图像中的风格特征对扩散模型进行后门攻击 cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.20650v2) [paper-pdf](http://arxiv.org/pdf/2502.20650v2)

**Authors**: Yu Pan, Bingrong Dai, Jiahao Chen, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.

摘要: 近年来，扩散模型（DM）在图像生成领域取得了重大进展。然而，根据当前的研究，DM很容易受到后门攻击，后门攻击允许攻击者通过输入包含隐蔽触发器（例如特定的视觉补丁或短语）的数据来控制模型的输出。现有的防御策略完全可以通过后门检测和触发器倒置来阻止此类攻击，因为以前的攻击方法受到有限的输入空间和低维触发器的限制。例如，视觉触发器很容易被防御者观察到，基于文本或基于注意力的触发器更容易受到神经网络检测的影响。为了探索DM中后门攻击的更多可能性，我们提出了Gungnir，这是一种新颖的方法，使攻击者能够通过输入图像中的风格触发器激活DM中的后门。我们的方法首次提出使用风格特征作为触发器，并通过引入重建对抗噪音（RAN）和短期时间间隔保留（STTR）在图像到图像任务中成功实施后门攻击。我们的技术生成的嵌入式图像在感知上与干净图像无法区分，从而绕过了手动检查和自动检测神经网络。实验表明，贡尼尔可以轻松绕过现有的防御方法。在现有的DM防御框架中，我们的方法实现了0后门检测率（BDR）。我们的代码可在https://github.com/paoche11/Gungnir上获取。



## **9. Evaluating the Robustness of Multimodal Agents Against Active Environmental Injection Attacks**

多模态Agent对主动环境注入攻击的鲁棒性评估 cs.CL

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.13053v2) [paper-pdf](http://arxiv.org/pdf/2502.13053v2)

**Authors**: Yurun Chen, Xavier Hu, Keting Yin, Juncheng Li, Shengyu Zhang

**Abstract**: As researchers continue to optimize AI agents for more effective task execution within operating systems, they often overlook a critical security concern: the ability of these agents to detect "impostors" within their environment. Through an analysis of the agents' operational context, we identify a significant threat-attackers can disguise malicious attacks as environmental elements, injecting active disturbances into the agents' execution processes to manipulate their decision-making. We define this novel threat as the Active Environment Injection Attack (AEIA). Focusing on the interaction mechanisms of the Android OS, we conduct a risk assessment of AEIA and identify two critical security vulnerabilities: (1) Adversarial content injection in multimodal interaction interfaces, where attackers embed adversarial instructions within environmental elements to mislead agent decision-making; and (2) Reasoning gap vulnerabilities in the agent's task execution process, which increase susceptibility to AEIA attacks during reasoning. To evaluate the impact of these vulnerabilities, we propose AEIA-MN, an attack scheme that exploits interaction vulnerabilities in mobile operating systems to assess the robustness of MLLM-based agents. Experimental results show that even advanced MLLMs are highly vulnerable to this attack, achieving a maximum attack success rate of 93% on the AndroidWorld benchmark by combining two vulnerabilities.

摘要: 随着研究人员不断优化人工智能代理，以便在操作系统中更有效地执行任务，他们往往忽视了一个关键的安全问题：这些代理在其环境中检测“冒名顶替者”的能力。通过分析代理的操作环境，我们确定了一个重大的威胁-攻击者可以伪装成环境元素的恶意攻击，注入主动干扰代理的执行过程，操纵他们的决策。我们将这种新的威胁定义为主动环境注入攻击（AEIA）。针对Android操作系统的交互机制，我们对AEIA进行了风险评估，并发现了两个关键的安全漏洞：（1）多模态交互界面中的对抗性内容注入，攻击者在环境元素中嵌入对抗性指令，以误导Agent决策;和（2）代理任务执行过程中的推理缺口漏洞，这增加了推理期间对AEIA攻击的易感性。为了评估这些漏洞的影响，我们提出了AEIA-NN，这是一种攻击方案，利用移动操作系统中的交互漏洞来评估基于MLLM的代理的稳健性。实验结果表明，即使是高级MLLM也极易受到这种攻击，通过组合两个漏洞，在AndroidWorld基准上实现了93%的最高攻击成功率。



## **10. Unifying Image Counterfactuals and Feature Attributions with Latent-Space Adversarial Attacks**

统一图像反事实和特征属性与潜在空间对抗攻击 cs.LG

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15479v1) [paper-pdf](http://arxiv.org/pdf/2504.15479v1)

**Authors**: Jeremy Goldwasser, Giles Hooker

**Abstract**: Counterfactuals are a popular framework for interpreting machine learning predictions. These what if explanations are notoriously challenging to create for computer vision models: standard gradient-based methods are prone to produce adversarial examples, in which imperceptible modifications to image pixels provoke large changes in predictions. We introduce a new, easy-to-implement framework for counterfactual images that can flexibly adapt to contemporary advances in generative modeling. Our method, Counterfactual Attacks, resembles an adversarial attack on the representation of the image along a low-dimensional manifold. In addition, given an auxiliary dataset of image descriptors, we show how to accompany counterfactuals with feature attribution that quantify the changes between the original and counterfactual images. These importance scores can be aggregated into global counterfactual explanations that highlight the overall features driving model predictions. While this unification is possible for any counterfactual method, it has particular computational efficiency for ours. We demonstrate the efficacy of our approach with the MNIST and CelebA datasets.

摘要: 反事实是解释机器学习预测的流行框架。众所周知，为计算机视觉模型创建这些假设解释具有挑战性：标准的基于梯度的方法容易产生对抗性的例子，其中对图像像素的难以察觉的修改会引发预测的巨大变化。我们为反事实图像引入了一种新的、易于实施的框架，该框架可以灵活地适应生成建模的当代进步。我们的方法“反事实攻击”类似于对沿着低维多管的图像表示的对抗攻击。此外，给定图像描述符的辅助数据集，我们展示了如何伴随反事实与特征属性，量化原始图像和反事实图像之间的变化。这些重要性分数可以聚合成全局反事实解释，突出显示驱动模型预测的整体特征。虽然这种统一对任何反事实方法都是可能的，但它对我们的方法具有特定的计算效率。我们用MNIST和CelebA数据集证明了我们方法的有效性。



## **11. An Undetectable Watermark for Generative Image Models**

生成图像模型的不可检测水印 cs.CR

ICLR 2025

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.07369v4) [paper-pdf](http://arxiv.org/pdf/2410.07369v4)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.

摘要: 我们提出了第一个用于生成图像模型的不可检测水印方案。不可检测性确保没有有效的对手能够区分带水印和未带水印的图像，即使在进行了多次自适应查询之后。特别是，在任何有效可计算的指标下，不可检测的水印不会降低图像质量。我们的方案通过使用伪随机错误纠正码（Christ和Gunn，2024）选择扩散模型的初始潜伏，这是一种保证不可检测性和鲁棒性的策略。我们通过实验证明，使用稳定扩散2.1，我们的水印具有质量保护性和鲁棒性。我们的实验证明，与我们测试的每个先前方案相比，我们的水印不会降低图像质量。我们的实验还证明了鲁棒性：现有的水印去除攻击无法在不显着降低图像质量的情况下从图像中去除水印。最后，我们发现我们可以对水印中的512位进行鲁棒性编码，当图像不受到水印删除攻击时，最多可编码2500位。我们的代码可在https://github.com/XuandongZhao/PRC-Watermark上获取。



## **12. A Framework for Evaluating Emerging Cyberattack Capabilities of AI**

评估人工智能新兴网络攻击能力的框架 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.11917v3) [paper-pdf](http://arxiv.org/pdf/2503.11917v3)

**Authors**: Mikel Rodriguez, Raluca Ada Popa, Four Flynn, Lihao Liang, Allan Dafoe, Anna Wang

**Abstract**: As frontier AI models become more capable, evaluating their potential to enable cyberattacks is crucial for ensuring the safe development of Artificial General Intelligence (AGI). Current cyber evaluation efforts are often ad-hoc, lacking systematic analysis of attack phases and guidance on targeted defenses. This work introduces a novel evaluation framework that addresses these limitations by: (1) examining the end-to-end attack chain, (2) identifying gaps in AI threat evaluation, and (3) helping defenders prioritize targeted mitigations and conduct AI-enabled adversary emulation for red teaming. Our approach adapts existing cyberattack chain frameworks for AI systems. We analyzed over 12,000 real-world instances of AI involvement in cyber incidents, catalogued by Google's Threat Intelligence Group, to curate seven representative attack chain archetypes. Through a bottleneck analysis on these archetypes, we pinpointed phases most susceptible to AI-driven disruption. We then identified and utilized externally developed cybersecurity model evaluations focused on these critical phases. We report on AI's potential to amplify offensive capabilities across specific attack stages, and offer recommendations for prioritizing defenses. We believe this represents the most comprehensive AI cyber risk evaluation framework published to date.

摘要: 随着前沿人工智能模型变得越来越强大，评估其实现网络攻击的潜力对于确保通用人工智能（AGI）的安全发展至关重要。目前的网络评估工作往往是临时性的，缺乏对攻击阶段的系统分析和对有针对性防御的指导。这项工作引入了一个新的评估框架，通过以下方式解决这些限制：（1）检查端到端攻击链，（2）识别AI威胁评估中的差距，以及（3）帮助防御者优先考虑有针对性的缓解措施，并进行支持AI的对手模拟。我们的方法将现有的网络攻击链框架应用于人工智能系统。我们分析了谷歌威胁情报小组编目的12，000多个人工智能参与网络事件的现实案例，以策划七个有代表性的攻击链原型。通过对这些原型的瓶颈分析，我们找到了最容易受到人工智能驱动破坏的阶段。然后，我们确定并利用外部开发的专注于这些关键阶段的网络安全模型评估。我们报告了人工智能在特定攻击阶段增强进攻能力的潜力，并提供了优先考虑防御的建议。我们相信这代表了迄今为止发布的最全面的人工智能网络风险评估框架。



## **13. MST3 Encryption improvement with three-parameter group of Hermitian function field**

MST 3使用Hermitian函数域的三参数组进行加密改进 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15391v1) [paper-pdf](http://arxiv.org/pdf/2504.15391v1)

**Authors**: Gennady Khalimov, Yevgen Kotukh

**Abstract**: This scholarly work presents an advanced cryptographic framework utilizing automorphism groups as the foundational structure for encryption scheme implementation. The proposed methodology employs a three-parameter group construction, distinguished by its application of logarithmic signatures positioned outside the group's center, a significant departure from conventional approaches. A key innovation in this implementation is utilizing the Hermitian function field as the underlying mathematical framework. This particular function field provides enhanced structural properties that strengthen the cryptographic protocol when integrated with the three-parameter group architecture. The encryption mechanism features phased key de-encapsulation from ciphertext, representing a substantial advantage over alternative implementations. This sequential extraction process introduces additional computational complexity for potential adversaries while maintaining efficient legitimate decryption. A notable characteristic of this cryptosystem is the direct correlation between the underlying group's mathematical strength and both the attack complexity and message size parameters. This relationship enables precise security-efficiency calibration based on specific implementation requirements and threat models. The application of automorphism groups with logarithmic signatures positioned outside the center represents a significant advancement in non-traditional cryptographic designs, particularly relevant in the context of post-quantum cryptographic resilience.

摘要: 这项学术工作提出了一个先进的加密框架，利用自同构群作为加密方案实现的基础结构。所提出的方法采用了三参数组的建设，其应用对数签名定位组的中心之外，显着偏离传统的方法。在这种实现中的一个关键创新是利用厄米函数场作为底层的数学框架。这个特定的函数字段提供增强的结构属性，当与三参数组架构集成时，该结构属性增强了加密协议。该加密机制的特点是分阶段从密文中解封装密钥，与替代实现相比具有巨大优势。这种顺序提取过程为潜在对手带来了额外的计算复杂性，同时保持高效的合法解密。该加密系统的一个显着特征是基础组的数学强度与攻击复杂性和消息大小参数之间直接相关。这种关系可以根据特定的实施要求和威胁模型进行精确的安全效率校准。对数签名位于中心之外的自同构群的应用代表了非传统密码设计的重大进步，特别是在后量子密码弹性的背景下。



## **14. MR. Guard: Multilingual Reasoning Guardrail using Curriculum Learning**

Mr. Guard：使用课程学习的多语言推理保护 cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15241v1) [paper-pdf](http://arxiv.org/pdf/2504.15241v1)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual setting, where multilingual safety-aligned data are often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we propose an approach to build a multilingual guardrail with reasoning. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-guided Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail consistently outperforms recent baselines across both in-domain and out-of-domain languages. The multilingual reasoning capability of our guardrail enables it to generate multilingual explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.

摘要: 大型语言模型（LLM）容易受到诸如越狱之类的对抗性攻击，这可能会引发有害或不安全的行为。在多语言环境中，这种漏洞会加剧，因为多语言安全一致的数据通常有限。因此，开发一个能够检测和过滤不同语言的不安全内容的护栏对于在现实世界的应用程序中部署LLM至关重要。在这项工作中，我们提出了一种方法来建立一个多语言护栏推理。我们的方法包括：（1）综合多语言数据生成，融合了文化和语言上的细微差别，（2）监督微调，以及（3）课程引导的群组相对政策优化（GRPO）框架，进一步提高性能。实验结果表明，我们的多语言护栏在域内和域外语言中始终优于最近的基线。我们护栏的多语言推理能力使其能够生成多语言解释，这对于理解多语言内容审核中的特定语言风险和歧义特别有用。



## **15. Progressive Pruning: Analyzing the Impact of Intersection Attacks**

渐进修剪：分析交叉点攻击的影响 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.08700v2) [paper-pdf](http://arxiv.org/pdf/2410.08700v2)

**Authors**: Christoph Döpmann, Maximilian Weisenseel, Florian Tschorsch

**Abstract**: Stream-based communication dominates today's Internet, posing unique challenges for anonymous communication networks (ACNs). Traditionally designed for independent messages, ACNs struggle to account for the inherent vulnerabilities of streams, such as susceptibility to intersection attacks. In this work, we address this gap and introduce progressive pruning, a novel methodology for quantifying the susceptibility to intersection attacks. Progressive pruning quantifies and monitors anonymity sets over time, providing an assessment of an adversary's success in correlating senders and receivers. We leverage this methodology to analyze synthetic scenarios and large-scale simulations of the Tor network using our newly developed TorFS simulator. Our findings reveal that anonymity is significantly influenced by stream length, user population, and stream distribution across the network. These insights highlight critical design challenges for future ACNs seeking to safeguard stream-based communication against traffic analysis attacks.

摘要: 基于流的通信在当今的互联网中占据主导地位，给匿名通信网络（ACN）带来了独特的挑战。ACN传统上是为独立消息设计的，很难解决流的固有漏洞，例如容易受到交叉攻击。在这项工作中，我们解决了这一差距并引入了渐进式修剪，这是一种用于量化交叉攻击易感性的新颖方法。渐进式修剪随着时间的推移量化和监控匿名集，提供对对手关联收件箱和接收器的成功评估。我们利用这种方法来使用我们新开发的TorFS模拟器分析Tor网络的合成场景和大规模模拟。我们的研究结果表明，匿名性受到流长度、用户群体和整个网络的流分布的显着影响。这些见解凸显了未来ACN的关键设计挑战，这些ACN旨在保护基于流的通信免受流量分析攻击。



## **16. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

HiddenDetect：通过监视隐藏状态检测针对大型视觉语言模型的越狱攻击 cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.14744v3) [paper-pdf](http://arxiv.org/pdf/2502.14744v3)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.

摘要: 与纯语言模型相比，其他模式的集成增加了大型视觉语言模型（LVLM）对安全风险（如越狱攻击）的敏感性。虽然现有的研究主要集中在事后对齐技术，LVLM内的潜在安全机制仍然在很大程度上未被探索。在这项工作中，我们调查是否LVLM内在编码安全相关的信号在其内部激活过程中的推理。我们的研究结果表明，LVLM在处理不安全的提示时表现出不同的激活模式，可以利用它来检测和减轻对抗性输入，而不需要进行广泛的微调。基于这一见解，我们引入了HiddenDetect，这是一个新颖的免调框架，可以利用内部模型激活来增强安全性。实验结果表明，{HiddenDetect}在检测针对LVLM的越狱攻击方面超越了最先进的方法。通过利用固有的安全感知模式，我们的方法提供了一种高效且可扩展的解决方案，用于加强LVLM针对多模式威胁的鲁棒性。我们的代码将在https://github.com/leigest519/HiddenDetect上公开发布。



## **17. Scalable Discrete Event Simulation Tool for Large-Scale Cyber-Physical Energy Systems: Advancing System Efficiency and Scalability**

用于大规模网络物理能源系统的可扩展离散事件模拟工具：提高系统效率和可扩展性 eess.SY

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15198v1) [paper-pdf](http://arxiv.org/pdf/2504.15198v1)

**Authors**: Khandaker Akramul Haque, Shining Sun, Xiang Huo, Ana E. Goulart, Katherine R. Davis

**Abstract**: Modern power systems face growing risks from cyber-physical attacks, necessitating enhanced resilience due to their societal function as critical infrastructures. The challenge is that defense of large-scale systems-of-systems requires scalability in their threat and risk assessment environment for cyber physical analysis including cyber-informed transmission planning, decision-making, and intrusion response. Hence, we present a scalable discrete event simulation tool for analysis of energy systems, called DESTinE. The tool is tailored for largescale cyber-physical systems, with a focus on power systems. It supports faster-than-real-time traffic generation and models packet flow and congestion under both normal and adversarial conditions. Using three well-established power system synthetic cases with 500, 2000, and 10,000 buses, we overlay a constructed cyber network employing star and radial topologies. Experiments are conducted to identify critical nodes within a communication network in response to a disturbance. The findings are incorporated into a constrained optimization problem to assess the impact of the disturbance on a specific node and its cascading effects on the overall network. Based on the solution of the optimization problem, a new hybrid network topology is also derived, combining the strengths of star and radial structures to improve network resilience. Furthermore, DESTinE is integrated with a virtual server and a hardware-in-the-loop (HIL) system using Raspberry Pi 5.

摘要: 现代电力系统面临着越来越大的网络物理攻击风险，由于其作为关键基础设施的社会功能，因此需要增强弹性。挑战在于，大规模系统的防御需要其威胁和风险评估环境的可扩展性，以进行网络物理分析，包括基于网络的传输规划、决策和入侵响应。因此，我们提出了一种用于分析能源系统的可扩展离散事件模拟工具，称为DESTinE。该工具专为大型网络物理系统量身定制，重点关注电力系统。它支持比实时更快的流量生成，并对正常和对抗条件下的分组流和拥堵进行建模。我们使用三个成熟的电力系统合成案例（包含500、2000和10，000条公交车），覆盖了采用星型和放射状布局的构建网络网络。进行实验来识别通信网络内的关键节点以响应干扰。将研究结果纳入约束优化问题中，以评估干扰对特定节点的影响及其对整个网络的级联影响。在解决优化问题的基础上，还推导出了一种新的混合网络布局，结合星型和辐射型结构的优点，提高网络弹性。此外，DESTinE还使用Raspberry Pi 5与虚拟服务器和硬件在环（HIL）系统集成。



## **18. RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search**

RainbowPlus：通过进化质量多样性搜索增强对抗提示生成 cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15047v1) [paper-pdf](http://arxiv.org/pdf/2504.15047v1)

**Authors**: Quy-Anh Dang, Chris Ngo, Truong-Son Hy

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but are susceptible to adversarial prompts that exploit vulnerabilities to produce unsafe or biased outputs. Existing red-teaming methods often face scalability challenges, resource-intensive requirements, or limited diversity in attack strategies. We propose RainbowPlus, a novel red-teaming framework rooted in evolutionary computation, enhancing adversarial prompt generation through an adaptive quality-diversity (QD) search that extends classical evolutionary algorithms like MAP-Elites with innovations tailored for language models. By employing a multi-element archive to store diverse high-quality prompts and a comprehensive fitness function to evaluate multiple prompts concurrently, RainbowPlus overcomes the constraints of single-prompt archives and pairwise comparisons in prior QD methods like Rainbow Teaming. Experiments comparing RainbowPlus to QD methods across six benchmark datasets and four open-source LLMs demonstrate superior attack success rate (ASR) and diversity (Diverse-Score $\approx 0.84$), generating up to 100 times more unique prompts (e.g., 10,418 vs. 100 for Ministral-8B-Instruct-2410). Against nine state-of-the-art methods on the HarmBench dataset with twelve LLMs (ten open-source, two closed-source), RainbowPlus achieves an average ASR of 81.1%, surpassing AutoDAN-Turbo by 3.9%, and is 9 times faster (1.45 vs. 13.50 hours). Our open-source implementation fosters further advancements in LLM safety, offering a scalable tool for vulnerability assessment. Code and resources are publicly available at https://github.com/knoveleng/rainbowplus, supporting reproducibility and future research in LLM red-teaming.

摘要: 大型语言模型（LLM）表现出非凡的能力，但很容易受到对抗提示的影响，这些提示利用漏洞产生不安全或有偏见的输出。现有的红色团队方法通常面临可扩展性挑战、资源密集型要求或攻击策略的多样性有限。我们提出RainbowPlus，这是一种植根于进化计算的新型红色团队框架，通过自适应质量多样性（QD）搜索来增强对抗提示生成，该搜索扩展了MAP-Elites等经典进化算法，并为语言模型量身定制的创新。通过采用多元素档案来存储各种高质量提示，并采用全面的适应度函数来同时评估多个提示，RainbowPlus克服了Rainbow Teaming等现有QD方法中单提示档案和成对比较的限制。在六个基准数据集和四个开源LLM上比较RainbowPlus与QD方法的实验证明了卓越的攻击成功率（ASB）和多样性（Diverse-Score $\大约0.84$），生成高达100倍的独特提示（例如，Ministral-8B-Direct-2410为10，418 vs 100）。与HarmBench数据集中的九种最先进方法（12个LLM（10个开源，2个封闭源）相比，RainbowPlus的平均ASB为81.1%，超过AutoDAN-Turbo 3.9%，速度快9倍（1.45小时vs 13.50小时）。我们的开源实施促进了LLM安全性的进一步进步，为漏洞评估提供了可扩展的工具。代码和资源可在https：//github.com/knoveleng/rainbowplus上公开获取，支持LLM红色团队的再现性和未来研究。



## **19. An Information-theoretic Security Analysis of Honeyword**

蜜语的信息论安全分析 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2311.10960v2) [paper-pdf](http://arxiv.org/pdf/2311.10960v2)

**Authors**: Pengcheng Su, Haibo Cheng, Wenting Li, Ping Wang

**Abstract**: Honeyword is a representative "honey" technique that employs decoy objects to mislead adversaries and protect the real ones. To assess the security of a Honeyword system, two metrics--flatness and success-number--have been proposed and evaluated using various simulated attackers. Existing evaluations typically apply statistical learning methods to distinguish real passwords from decoys on real-world datasets. However, such evaluations may overestimate the system's security, as more effective distinguishing attacks could potentially exist.   In this paper, we aim to analyze the security of Honeyword systems under the strongest theoretical attack, rather than relying on specific, expert-crafted attacks evaluated in prior experimental studies. We first derive mathematical expressions for the flatness and success-number under the strongest attack. We conduct analyses and computations for several typical scenarios, and determine the security of honeyword generation methods using a uniform distribution and the List model as examples.   We further evaluate the security of existing honeyword generation methods based on password probability models (PPMs), which depends on the sample size used for training. We investigate, for the first time, the sample complexity of several representative PPMs, introducing two novel polynomial-time approximation schemes for computing the total variation between PCFG models and between higher-order Markov models. Our experimental results show that for small-scale password distributions, sample sizes on the order of millions--often tens of millions--are required to reduce the total variation below 0.1. A surprising result is that we establish an equivalence between flatness and total variation, thus bridging the theoretical study of Honeyword systems with classical information theory. Finally, we discuss the practical implications of our findings.

摘要: 蜜语是一种典型的“蜂蜜”技术，它利用诱饵对象来误导攻击者并保护真实对象。为了评估一个Honeyword系统的安全性，两个指标-平坦度和成功数-已被提出，并使用各种模拟攻击者进行评估。现有的评估通常应用统计学习方法来区分真实世界数据集上的真实密码和诱饵。然而，这种评估可能会高估系统的安全性，因为可能存在更有效的区分攻击。   在本文中，我们的目标是分析最强理论攻击下的Honeyword系统的安全性，而不是依赖于在先前的实验研究中评估的特定的、专家制作的攻击。我们首先推导出最强攻击下的平坦性和成功数的数学表达式。针对几种典型的情况进行了分析和计算，并以均匀分布和List模型为例，确定了蜜词生成方法的安全性。   我们进一步评估现有的基于密码概率模型（PPM），这取决于用于训练的样本大小的蜜词生成方法的安全性。我们首次研究了几种有代表性的PPP的样本复杂性，引入了两种新型的多项时间逼近方案来计算PCFG模型之间和更高次Markov模型之间的总方差。我们的实验结果表明，对于小规模密码分发，需要数百万（通常是数千万）数量级的样本量才能将总变异减少到0.1以下。一个令人惊讶的结果是，我们建立了平坦性和总变异性之间的等效性，从而将蜜语系统的理论研究与经典信息论联系起来。最后，我们讨论了我们发现的实际影响。



## **20. Transferable Adversarial Attacks on SAM and Its Downstream Models**

对Sam及其下游模型的可转移对抗攻击 cs.LG

update fig 1

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.20197v3) [paper-pdf](http://arxiv.org/pdf/2410.20197v3)

**Authors**: Song Xia, Wenhan Yang, Yi Yu, Xun Lin, Henghui Ding, Ling-Yu Duan, Xudong Jiang

**Abstract**: The utilization of large foundational models has a dilemma: while fine-tuning downstream tasks from them holds promise for making use of the well-generalized knowledge in practical applications, their open accessibility also poses threats of adverse usage. This paper, for the first time, explores the feasibility of adversarial attacking various downstream models fine-tuned from the segment anything model (SAM), by solely utilizing the information from the open-sourced SAM. In contrast to prevailing transfer-based adversarial attacks, we demonstrate the existence of adversarial dangers even without accessing the downstream task and dataset to train a similar surrogate model. To enhance the effectiveness of the adversarial attack towards models fine-tuned on unknown datasets, we propose a universal meta-initialization (UMI) algorithm to extract the intrinsic vulnerability inherent in the foundation model, which is then utilized as the prior knowledge to guide the generation of adversarial perturbations. Moreover, by formulating the gradient difference in the attacking process between the open-sourced SAM and its fine-tuned downstream models, we theoretically demonstrate that a deviation occurs in the adversarial update direction by directly maximizing the distance of encoded feature embeddings in the open-sourced SAM. Consequently, we propose a gradient robust loss that simulates the associated uncertainty with gradient-based noise augmentation to enhance the robustness of generated adversarial examples (AEs) towards this deviation, thus improving the transferability. Extensive experiments demonstrate the effectiveness of the proposed universal meta-initialized and gradient robust adversarial attack (UMI-GRAT) toward SAMs and their downstream models. Code is available at https://github.com/xiasong0501/GRAT.

摘要: 大型基础模型的利用面临一个困境：虽然对它们的下游任务进行微调有望在实际应用中利用良好的概括性知识，但它们的开放可访问性也构成了不良使用的威胁。本文首次探讨了对抗攻击从细分任何模型（Sam）微调的各种下游模型的可行性，仅利用开源Sam的信息。与流行的基于传输的对抗攻击相反，我们证明了对抗危险的存在，即使不访问下游任务和数据集来训练类似的代理模型。为了提高针对在未知数据集上微调的模型的对抗攻击的有效性，我们提出了一种通用元初始化（UMI）算法来提取基础模型中固有的内在漏洞，然后将其用作先验知识来指导对抗性扰动的生成。此外，通过公式化开放源代码的Sam及其微调下游模型之间攻击过程中的梯度差异，我们从理论上证明了通过直接最大化开放源代码的特征嵌入的距离，对抗性更新方向上会发生偏差。因此，我们提出了一种梯度鲁棒性损失，通过基于梯度的噪音增强来模拟相关的不确定性，以增强生成的对抗性示例（AE）对该偏差的鲁棒性，从而提高可移植性。大量实验证明了所提出的通用元初始化和梯度鲁棒对抗攻击（UMI-GRAT）对自组装体及其下游模型的有效性。代码可在www.example.com上获取。



## **21. aiXamine: LLM Safety and Security Simplified**

aiXamine：LLM安全与安保简化 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14985v1) [paper-pdf](http://arxiv.org/pdf/2504.14985v1)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.

摘要: 评估大型语言模型（LLM）的安全性和保障性仍然是一项复杂的任务，通常需要用户在临时基准、数据集、指标和报告格式的碎片化环境中进行导航。为了应对这一挑战，我们推出了aiXamine，这是一个针对LLM安全性的全面黑匣子评估平台。aiXamine集成了40多个测试（即，基准）组织成八个关键服务，针对安全和保障的特定维度：对抗稳健性、代码安全性、公平性和偏见、幻觉、模型和数据隐私、分发外（OOD）稳健性、过度拒绝和安全对齐。该平台将评估结果汇总到每个模型的单个详细报告中，提供模型性能、测试示例和丰富的可视化的详细细分。我们使用aiXamine评估了50多个公开和专有的LLM，进行了超过2000次检查。我们的研究结果揭示了领先模型中的显着漏洞，包括OpenAI GPT-4 o中容易受到对抗攻击、xAI Grok-3中的偏见输出以及Google Gemini 2.0中的隐私弱点。此外，我们观察到开源模型可以在特定服务中匹配或超过专有模型，例如安全性一致、公平性和偏差以及OOD稳健性。最后，我们确定了蒸馏策略、模型大小、训练方法和架构选择之间的权衡。



## **22. Fast Adversarial Training with Weak-to-Strong Spatial-Temporal Consistency in the Frequency Domain on Videos**

视频频域中具有弱到强时空一致性的快速对抗训练 cs.CV

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14921v1) [paper-pdf](http://arxiv.org/pdf/2504.14921v1)

**Authors**: Songping Wang, Hanqing Liu, Yueming Lyu, Xiantao Hu, Ziwen He, Wei Wang, Caifeng Shan, Liang Wang

**Abstract**: Adversarial Training (AT) has been shown to significantly enhance adversarial robustness via a min-max optimization approach. However, its effectiveness in video recognition tasks is hampered by two main challenges. First, fast adversarial training for video models remains largely unexplored, which severely impedes its practical applications. Specifically, most video adversarial training methods are computationally costly, with long training times and high expenses. Second, existing methods struggle with the trade-off between clean accuracy and adversarial robustness. To address these challenges, we introduce Video Fast Adversarial Training with Weak-to-Strong consistency (VFAT-WS), the first fast adversarial training method for video data. Specifically, VFAT-WS incorporates the following key designs: First, it integrates a straightforward yet effective temporal frequency augmentation (TF-AUG), and its spatial-temporal enhanced form STF-AUG, along with a single-step PGD attack to boost training efficiency and robustness. Second, it devises a weak-to-strong spatial-temporal consistency regularization, which seamlessly integrates the simpler TF-AUG and the more complex STF-AUG. Leveraging the consistency regularization, it steers the learning process from simple to complex augmentations. Both of them work together to achieve a better trade-off between clean accuracy and robustness. Extensive experiments on UCF-101 and HMDB-51 with both CNN and Transformer-based models demonstrate that VFAT-WS achieves great improvements in adversarial robustness and corruption robustness, while accelerating training by nearly 490%.

摘要: 对抗训练（AT）已被证明可以通过最小-最大优化方法显着增强对抗鲁棒性。然而，它在视频识别任务中的有效性受到两个主要挑战的阻碍。首先，视频模型的快速对抗训练在很大程度上尚未开发，这严重阻碍了其实际应用。具体来说，大多数视频对抗训练方法的计算成本很高，训练时间长，费用高。其次，现有的方法很难在清晰的准确性和对抗性稳健性之间做出权衡。为了应对这些挑战，我们引入了具有弱到强一致性的视频快速对抗训练（VFAT-WS），这是第一种针对视频数据的快速对抗训练方法。具体来说，VFAT-WS结合了以下关键设计：首先，它集成了简单而有效的时间频率增强（TF-AUG）及其时空增强形式STF-AUG，以及一步PVD攻击，以提高训练效率和鲁棒性。其次，它设计了从弱到强的时空一致性规范化，无缝集成了更简单的TF-AUG和更复杂的STF-AUG。利用一致性规范化，它将学习过程从简单扩展转向复杂扩展。两者共同努力，在干净的准确性和稳健性之间实现更好的权衡。使用CNN和基于Transformer的模型在UCF-101和HMDB-51上进行的广泛实验表明，VFAT-WS在对抗稳健性和腐败稳健性方面实现了巨大改进，同时将训练速度加快了近490%。



## **23. PA-Boot: A Formally Verified Authentication Protocol for Multiprocessor Secure Boot**

PA-Boot：用于多处理器安全引导的正式验证认证协议 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2209.07936v3) [paper-pdf](http://arxiv.org/pdf/2209.07936v3)

**Authors**: Zhuoruo Zhang, Rui Chang, Mingshuai Chen, Wenbo Shen, Chenyang Yu, He Huang, Qinming Dai, Yongwang Zhao

**Abstract**: Hardware supply-chain attacks are raising significant security threats to the boot process of multiprocessor systems. This paper identifies a new, prevalent hardware supply-chain attack surface that can bypass multiprocessor secure boot due to the absence of processor-authentication mechanisms. To defend against such attacks, we present PA-Boot, the first formally verified processor-authentication protocol for secure boot in multiprocessor systems. PA-Boot is proved functionally correct and is guaranteed to detect multiple adversarial behaviors, e.g., processor replacements, man-in-the-middle attacks, and tampering with certificates. The fine-grained formalization of PA-Boot and its fully mechanized security proofs are carried out in the Isabelle/HOL theorem prover with 306 lemmas/theorems and ~7,100 LoC. Experiments on a proof-of-concept implementation indicate that PA-Boot can effectively identify boot-process attacks with a considerably minor overhead and thereby improve the security of multiprocessor systems.

摘要: 硬件供应链攻击正在对多处理器系统的引导过程构成重大安全威胁。本文指出了一种新的、流行的硬件供应链攻击面，由于缺乏处理器身份验证机制，该攻击面可以绕过多处理器的安全引导。为了抵御此类攻击，我们提出了PA-Boot，这是第一个经过正式验证的处理器身份验证协议，用于多处理器系统中的安全引导。PA-Boot被证明功能正确，并保证能够检测到多种对抗行为，例如，处理器更换、中间人攻击和篡改证书。PA-Boot的细粒度形式化及其完全机械化的安全证明是在Isabelle/HOL定理证明器中执行的，具有306个引理/定理和约7，100 LoC。概念验证实现的实验表明，PA-Boot可以以相当小的负担有效地识别引导进程攻击，从而提高多处理器系统的安全性。



## **24. Verifying Robust Unlearning: Probing Residual Knowledge in Unlearned Models**

稳健的非学习：探索未学习模型中的剩余知识 cs.LG

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14798v1) [paper-pdf](http://arxiv.org/pdf/2504.14798v1)

**Authors**: Hao Xuan, Xingyu Li

**Abstract**: Machine Unlearning (MUL) is crucial for privacy protection and content regulation, yet recent studies reveal that traces of forgotten information persist in unlearned models, enabling adversaries to resurface removed knowledge. Existing verification methods only confirm whether unlearning was executed, failing to detect such residual information leaks. To address this, we introduce the concept of Robust Unlearning, ensuring models are indistinguishable from retraining and resistant to adversarial recovery. To empirically evaluate whether unlearning techniques meet this security standard, we propose the Unlearning Mapping Attack (UMA), a post-unlearning verification framework that actively probes models for forgotten traces using adversarial queries. Extensive experiments on discriminative and generative tasks show that existing unlearning techniques remain vulnerable, even when passing existing verification metrics. By establishing UMA as a practical verification tool, this study sets a new standard for assessing and enhancing machine unlearning security.

摘要: 机器非学习（MUL）对于隐私保护和内容监管至关重要，但最近的研究表明，被遗忘的信息的痕迹仍然存在于未学习的模型中，使对手能够重新暴露被删除的知识。现有的验证方法仅确认是否执行了取消学习，无法检测到此类剩余信息泄露。为了解决这个问题，我们引入了稳健的非学习概念，确保模型与再培训没有区别，并且能够抵抗对抗性恢复。为了从经验上评估取消学习技术是否满足这一安全标准，我们提出了取消学习映射攻击（UMA），这是一个取消学习后验证框架，它使用对抗性查询主动探测模型中的遗忘痕迹。关于区分性和生成性任务的广泛实验表明，即使通过了现有的验证指标，现有的取消学习技术仍然很脆弱。通过将UMA建立为实用验证工具，本研究为评估和增强机器学习安全性设定了新标准。



## **25. Modality Unified Attack for Omni-Modality Person Re-Identification**

全方位人重新识别的情态统一攻击 cs.CV

9 pages,3 figures

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2501.12761v2) [paper-pdf](http://arxiv.org/pdf/2501.12761v2)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yunfeng Ma, Yaonan Wang

**Abstract**: Deep learning based person re-identification (re-id) models have been widely employed in surveillance systems. Recent studies have demonstrated that black-box single-modality and cross-modality re-id models are vulnerable to adversarial examples (AEs), leaving the robustness of multi-modality re-id models unexplored. Due to the lack of knowledge about the specific type of model deployed in the target black-box surveillance system, we aim to generate modality unified AEs for omni-modality (single-, cross- and multi-modality) re-id models. Specifically, we propose a novel Modality Unified Attack method to train modality-specific adversarial generators to generate AEs that effectively attack different omni-modality models. A multi-modality model is adopted as the surrogate model, wherein the features of each modality are perturbed by metric disruption loss before fusion. To collapse the common features of omni-modality models, Cross Modality Simulated Disruption approach is introduced to mimic the cross-modality feature embeddings by intentionally feeding images to non-corresponding modality-specific subnetworks of the surrogate model. Moreover, Multi Modality Collaborative Disruption strategy is devised to facilitate the attacker to comprehensively corrupt the informative content of person images by leveraging a multi modality feature collaborative metric disruption loss. Extensive experiments show that our MUA method can effectively attack the omni-modality re-id models, achieving 55.9%, 24.4%, 49.0% and 62.7% mean mAP Drop Rate, respectively.

摘要: 基于深度学习的人员重新识别（re-id）模型已广泛应用于监控系统中。最近的研究表明，黑匣子单模式和跨模式re-id模型容易受到对抗性示例（AE）的影响，因此多模式re-id模型的稳健性尚未得到探索。由于缺乏对目标黑匣子监控系统中部署的特定类型模型的了解，我们的目标是为全模式（单模式、交叉模式和多模式）re-id模型生成模式统一AE。具体来说，我们提出了一种新颖的模式统一攻击方法来训练特定于模式的对抗生成器，以生成有效攻击不同全模式模型的AE。采用多模式模型作为代理模型，其中每个模式的特征在融合之前受到度量中断损失的干扰。为了瓦解全模式模型的共同特征，引入了跨模式模拟破坏方法，通过有意地将图像提供给代理模型的非对应模式特定子网络来模拟跨模式特征嵌入。此外，多模式协同破坏策略旨在帮助攻击者通过利用多模式特征协同指标破坏损失来全面破坏人物图像的信息内容。大量实验表明，我们的MUA方法可以有效攻击全模式re-id模型，平均mAP下降率分别达到55.9%、24.4%、49.0%和62.7%。



## **26. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

展望未来：通过揭露对抗性合同来防止DeFi攻击 cs.CR

23 pages, 7 figures; Accepted to FSE 2025, to be published in The  Proceedings of the ACM on Software Engineering (PACMSE)

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2401.07261v6) [paper-pdf](http://arxiv.org/pdf/2401.07261v6)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: The exploitation of smart contract vulnerabilities in Decentralized Finance (DeFi) has resulted in financial losses exceeding 3 billion US dollars. Existing defense mechanisms primarily focus on detecting and reacting to adversarial transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively. Based on the fact that most attack logic rely on deploying intermediate smart contracts as supporting components to the exploitation of victim contracts, novel detection methods have been proposed that focus on identifying these adversarial contracts instead of adversarial transactions. However, previous state-of-the-art approaches in this direction have failed to produce results satisfactory enough for real-world deployment. In this paper, we propose LookAhead, a new framework for detecting DeFi attacks via unveiling adversarial contracts. LookAhead leverages common attack patterns, code semantics and intrinsic characteristics found in adversarial smart contracts to train Machine Learning (ML)-based classifiers that can effectively distinguish adversarial contracts from benign ones and make timely predictions of different types of potential attacks. Experiments on our labeled datasets show that LookAhead achieves an F1-score as high as 0.8966, which represents an improvement of over 44.4% compared to the previous state-of-the-art solution, with a False Positive Rate (FPR) at only 0.16%.

摘要: 利用去中心金融（DeFi）中的智能合约漏洞已造成超过30亿美元的财务损失。现有的防御机制主要专注于检测和反应攻击者针对受害者合同执行的对抗交易。然而，随着私有交易池的出现，交易直接发送给矿工，而无需首先出现在公共内存池中，当前的检测工具在有效识别攻击活动方面面临着重大挑战。基于大多数攻击逻辑依赖于部署中间智能合同作为利用受害者合同的支持组件这一事实，人们提出了新型检测方法，重点关注识别这些对抗性合同而不是对抗性交易。然而，之前在此方向上的最新方法未能产生足够令人满意的结果，以满足现实世界的部署。在本文中，我们提出了LookAhead，这是一个通过公布对抗性合同来检测DeFi攻击的新框架。LookAhead利用对抗性智能合同中的常见攻击模式、代码语义和内在特征来训练基于机器学习（ML）的分类器，可以有效区分对抗性合同与良性合同，并对不同类型的潜在攻击做出及时预测。对我们标记的数据集的实验表明，LookAhead的F1评分高达0.8966，与之前的最先进解决方案相比提高了44.4%以上，假阳性率（FPR）仅为0.16%。



## **27. Human-AI Collaboration in Cloud Security: Cognitive Hierarchy-Driven Deep Reinforcement Learning**

云安全中的人机协作：认知层次驱动的深度强化学习 cs.CR

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2502.16054v2) [paper-pdf](http://arxiv.org/pdf/2502.16054v2)

**Authors**: Zahra Aref, Sheng Wei, Narayan B. Mandayam

**Abstract**: Given the complexity of multi-tenant cloud environments and the growing need for real-time threat mitigation, Security Operations Centers (SOCs) must adopt AI-driven adaptive defense mechanisms to counter Advanced Persistent Threats (APTs). However, SOC analysts face challenges in handling adaptive adversarial tactics, requiring intelligent decision-support frameworks. We propose a Cognitive Hierarchy Theory-driven Deep Q-Network (CHT-DQN) framework that models interactive decision-making between SOC analysts and AI-driven APT bots. The SOC analyst (defender) operates at cognitive level-1, anticipating attacker strategies, while the APT bot (attacker) follows a level-0 policy. By incorporating CHT into DQN, our framework enhances adaptive SOC defense using Attack Graph (AG)-based reinforcement learning. Simulation experiments across varying AG complexities show that CHT-DQN consistently achieves higher data protection and lower action discrepancies compared to standard DQN. A theoretical lower bound further confirms its superiority as AG complexity increases. A human-in-the-loop (HITL) evaluation on Amazon Mechanical Turk (MTurk) reveals that SOC analysts using CHT-DQN-derived transition probabilities align more closely with adaptive attackers, leading to better defense outcomes. Moreover, human behavior aligns with Prospect Theory (PT) and Cumulative Prospect Theory (CPT): participants are less likely to reselect failed actions and more likely to persist with successful ones. This asymmetry reflects amplified loss sensitivity and biased probability weighting -- underestimating gains after failure and overestimating continued success. Our findings highlight the potential of integrating cognitive models into deep reinforcement learning to improve real-time SOC decision-making for cloud security.

摘要: 鉴于多租户云环境的复杂性以及对实时威胁缓解的日益增长的需求，安全运营中心（SOC）必须采用人工智能驱动的自适应防御机制来应对高级持续性威胁（APT）。然而，SOC分析师在处理自适应对抗策略方面面临挑战，需要智能的决策支持框架。我们提出了一个认知层次理论驱动的深度Q网络（CHT-DQN）框架，该框架对SOC分析师和人工智能驱动的APT机器人之间的交互决策进行建模。SOC分析师（防御者）在认知级别1下工作，预测攻击者策略，而APT机器人（攻击者）遵循0级策略。通过将CHT整合到DQN中，我们的框架使用基于攻击图（AG）的强化学习来增强自适应SOC防御。不同AG复杂性的模拟实验表明，与标准DQN相比，CHT-DQN始终实现更高的数据保护和更低的动作差异。随着AG复杂性的增加，理论下限进一步证实了其优势。对Amazon Mechanical Turk（MTurk）进行的人在环（HITL）评估显示，使用CHT-DQN推导的转移概率的SOC分析师与自适应攻击者更紧密地一致，从而获得更好的防御结果。此外，人类行为与前景理论（PT）和累积前景理论（CPD）一致：参与者不太可能重新选择失败的行动，而更有可能坚持成功的行动。这种不对称反映了损失敏感性的放大和概率加权的偏差--低估了失败后的收益并高估了持续成功。我们的研究结果强调了将认知模型集成到深度强化学习中以改善云安全的实时SOC决策的潜力。



## **28. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

大型语言模型作为软件分析中稳健的数据生成器：我们已经做到了吗？ cs.SE

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2411.10565v2) [paper-pdf](http://arxiv.org/pdf/2411.10565v2)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.

摘要: 大型语言模型（LLM）生成的数据越来越多地用于软件分析，但目前尚不清楚该数据与人类编写的数据相比如何，特别是当模型暴露于对抗场景时。对抗性攻击可能会损害软件系统的可靠性和安全性，因此，与作为模型性能基准的人类编写数据相比，了解LLM生成的数据在这些条件下的表现如何，可以为LLM生成的数据是否提供类似的稳健性和有效性提供有价值的见解。为了解决这一差距，我们系统地评估和比较人类编写的数据和LLM生成的数据的质量，以便在对抗性攻击的背景下微调稳健的预训练模型（Ptms）。我们评估了六种广泛使用的PtM的稳健性，这些PtM在对抗性攻击之前和之后根据人类编写和LLM生成的数据进行了微调。该评估在三个流行的软件分析任务中使用了九种最先进的（SOTA）对抗性攻击技术：克隆检测，代码摘要和代码审查讨论中的情感分析。此外，我们使用11个相似性度量来分析生成的对抗性示例的质量。我们的研究结果表明，虽然对LLM生成的数据进行微调的PTM与对人类编写的数据进行微调的PTM具有竞争力，但它们在软件分析任务中对对抗性攻击的鲁棒性较低。我们的研究强调了进一步探索提高LLM生成的训练数据质量的必要性，以开发高性能且能够抵御软件分析中的对抗性攻击的模型。



## **29. Towards Model Resistant to Transferable Adversarial Examples via Trigger Activation**

通过触发器激活建立抵抗可转移对抗示例的模型 cs.CR

Accepted by IEEE TIFS 2025

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2504.14541v1) [paper-pdf](http://arxiv.org/pdf/2504.14541v1)

**Authors**: Yi Yu, Song Xia, Xun Lin, Chenqi Kong, Wenhan Yang, Shijian Lu, Yap-Peng Tan, Alex C. Kot

**Abstract**: Adversarial examples, characterized by imperceptible perturbations, pose significant threats to deep neural networks by misleading their predictions. A critical aspect of these examples is their transferability, allowing them to deceive {unseen} models in black-box scenarios. Despite the widespread exploration of defense methods, including those on transferability, they show limitations: inefficient deployment, ineffective defense, and degraded performance on clean images. In this work, we introduce a novel training paradigm aimed at enhancing robustness against transferable adversarial examples (TAEs) in a more efficient and effective way. We propose a model that exhibits random guessing behavior when presented with clean data $\boldsymbol{x}$ as input, and generates accurate predictions when with triggered data $\boldsymbol{x}+\boldsymbol{\tau}$. Importantly, the trigger $\boldsymbol{\tau}$ remains constant for all data instances. We refer to these models as \textbf{models with trigger activation}. We are surprised to find that these models exhibit certain robustness against TAEs. Through the consideration of first-order gradients, we provide a theoretical analysis of this robustness. Moreover, through the joint optimization of the learnable trigger and the model, we achieve improved robustness to transferable attacks. Extensive experiments conducted across diverse datasets, evaluating a variety of attacking methods, underscore the effectiveness and superiority of our approach.

摘要: 以难以察觉的扰动为特征的对抗性示例通过误导深度神经网络的预测而对深度神经网络构成重大威胁。这些示例的一个关键方面是它们的可移植性，使它们能够在黑匣子场景中欺骗{看不见的}模型。尽管人们对防御方法（包括可移植性）进行了广泛的探索，但它们表现出局限性：部署效率低下、防御效率低下以及干净图像上的性能下降。在这项工作中，我们引入了一种新颖的训练范式，旨在以更高效、更有效的方式增强针对可转移对抗示例（TAE）的鲁棒性。我们提出了一个模型，当以干净的数据$\boldSymbol{x}$作为输入时，该模型表现出随机猜测行为，并在以触发的数据$\boldSymbol{x}+\boldSymbol{\tau}$作为输入时生成准确的预测。重要的是，触发器$\boldSymbol{\tau}$对于所有数据实例保持不变。我们将这些模型称为\textBF{具有触发器激活的模型}。我们惊讶地发现这些模型对TAE表现出一定的鲁棒性。通过考虑一阶梯度，我们对这种鲁棒性进行了理论分析。此外，通过可学习触发器和模型的联合优化，我们实现了对可转移攻击的更好的鲁棒性。在不同的数据集上进行了广泛的实验，评估了各种攻击方法，强调了我们方法的有效性和优越性。



## **30. Slice+Slice Baby: Generating Last-Level Cache Eviction Sets in the Blink of an Eye**

Slice+Slice Baby：眨眼间生成末级缓存驱逐集 cs.CR

Added reference to the ID3 decision tree induction algorithm by J. R.  Quinlan in Section 5.4

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2504.11208v2) [paper-pdf](http://arxiv.org/pdf/2504.11208v2)

**Authors**: Bradley Morgan, Gal Horowitz, Sioli O'Connell, Stephan van Schaik, Chitchanok Chuengsatiansup, Daniel Genkin, Olaf Maennel, Paul Montague, Eyal Ronen, Yuval Yarom

**Abstract**: An essential step for mounting cache attacks is finding eviction sets, collections of memory locations that contend on cache space. On Intel processors, one of the main challenges for identifying contending addresses is the sliced cache design, where the processor hashes the physical address to determine where in the cache a memory location is stored. While past works have demonstrated that the hash function can be reversed, they also showed that it depends on physical address bits that the adversary does not know.   In this work, we make three main contributions to the art of finding eviction sets. We first exploit microarchitectural races to compare memory access times and identify the cache slice to which an address maps. We then use the known hash function to both reduce the error rate in our slice identification method and to reduce the work by extrapolating slice mappings to untested memory addresses. Finally, we show how to propagate information on eviction sets across different page offsets for the hitherto unexplored case of non-linear hash functions.   Our contributions allow for entire LLC eviction set generation in 0.7 seconds on the Intel i7-9850H and 1.6 seconds on the i9-10900K, both using non-linear functions. This represents a significant improvement compared to state-of-the-art techniques taking 9x and 10x longer, respectively.

摘要: 发起缓存攻击的一个重要步骤是找到驱逐集，即争夺缓存空间的内存位置集合。在英特尔处理器上，识别竞争地址的主要挑战之一是切片高速缓存设计，其中处理器对物理地址进行哈希处理以确定内存位置存储在高速缓存中的位置。虽然过去的工作已经证明哈希函数可以颠倒，但他们也表明它取决于对手不知道的物理地址位。   在这部作品中，我们对寻找驱逐集的艺术做出了三项主要贡献。我们首先利用微体系结构的竞争比较内存访问时间，并确定地址映射到的缓存片。然后，我们使用已知的散列函数，以减少我们的切片识别方法中的错误率，并通过将切片映射外推到未经测试的内存地址来减少工作。最后，我们展示了如何传播信息驱逐集在不同的页面偏移量的非线性散列函数的情况下，迄今为止尚未探索。   我们的贡献允许在Intel i7- 9850 H上在0.7秒内生成整个LLC驱逐集，在i9- 10900 K上在1.6秒内生成整个LLC驱逐集，两者都使用非线性函数。与最先进的技术相比，这是一个显着的改进，分别耗时9倍和10倍。



## **31. Adversarial Attack for RGB-Event based Visual Object Tracking**

基于RGB事件的视觉对象跟踪的对抗攻击 cs.CV

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2504.14423v1) [paper-pdf](http://arxiv.org/pdf/2504.14423v1)

**Authors**: Qiang Chen, Xiao Wang, Haowen Wang, Bo Jiang, Lin Zhu, Dawei Zhang, Yonghong Tian, Jin Tang

**Abstract**: Visual object tracking is a crucial research topic in the fields of computer vision and multi-modal fusion. Among various approaches, robust visual tracking that combines RGB frames with Event streams has attracted increasing attention from researchers. While striving for high accuracy and efficiency in tracking, it is also important to explore how to effectively conduct adversarial attacks and defenses on RGB-Event stream tracking algorithms, yet research in this area remains relatively scarce. To bridge this gap, in this paper, we propose a cross-modal adversarial attack algorithm for RGB-Event visual tracking. Because of the diverse representations of Event streams, and given that Event voxels and frames are more commonly used, this paper will focus on these two representations for an in-depth study. Specifically, for the RGB-Event voxel, we first optimize the perturbation by adversarial loss to generate RGB frame adversarial examples. For discrete Event voxel representations, we propose a two-step attack strategy, more in detail, we first inject Event voxels into the target region as initialized adversarial examples, then, conduct a gradient-guided optimization by perturbing the spatial location of the Event voxels. For the RGB-Event frame based tracking, we optimize the cross-modal universal perturbation by integrating the gradient information from multimodal data. We evaluate the proposed approach against attacks on three widely used RGB-Event Tracking datasets, i.e., COESOT, FE108, and VisEvent. Extensive experiments show that our method significantly reduces the performance of the tracker across numerous datasets in both unimodal and multimodal scenarios. The source code will be released on https://github.com/Event-AHU/Adversarial_Attack_Defense

摘要: 视觉对象跟踪是计算机视觉和多模式融合领域的一个重要研究课题。在各种方法中，将Ruby帧与事件流相结合的鲁棒视觉跟踪引起了研究人员越来越多的关注。在努力实现跟踪的高准确性和高效性的同时，探索如何有效地对RGB-Events流跟踪算法进行对抗攻击和防御也很重要，但该领域的研究仍然相对较少。为了弥合这一差距，在本文中，我们提出了一种用于RGB-Events视觉跟踪的跨模式对抗攻击算法。由于事件流的表示方法多种多样，而事件体素和帧是比较常用的两种表示方法，本文将重点对这两种表示方法进行深入研究。具体来说，对于RGB事件体素，我们首先通过对抗性损失来优化扰动，以生成RGB帧对抗性示例。对于离散事件体素表示，我们提出了一种两步攻击策略，更详细地说，我们首先将事件体素注入目标区域作为初始化的对抗性示例，然后通过扰动事件体素的空间位置来进行梯度引导优化。对于基于RGB事件帧的跟踪，我们通过整合多模态数据的梯度信息来优化跨模态通用扰动。我们评估了针对三个广泛使用的RGB-事件跟踪数据集的攻击提出的方法，即COESOT、FE 108和VisEvents。大量实验表明，我们的方法在单模式和多模式场景中显着降低了跟踪器在众多数据集中的性能。源代码将在https://github.com/Event-AHU/Adversarial_Attack_Defense上发布



## **32. Hydra: An Agentic Reasoning Approach for Enhancing Adversarial Robustness and Mitigating Hallucinations in Vision-Language Models**

Hydra：一种增强视觉语言模型中对抗稳健性和减轻幻觉的抽象推理方法 cs.CV

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2504.14395v1) [paper-pdf](http://arxiv.org/pdf/2504.14395v1)

**Authors**: Chung-En, Yu, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian

**Abstract**: To develop trustworthy Vision-Language Models (VLMs), it is essential to address adversarial robustness and hallucination mitigation, both of which impact factual accuracy in high-stakes applications such as defense and healthcare. Existing methods primarily focus on either adversarial defense or hallucination post-hoc correction, leaving a gap in unified robustness strategies. We introduce \textbf{Hydra}, an adaptive agentic framework that enhances plug-in VLMs through iterative reasoning, structured critiques, and cross-model verification, improving both resilience to adversarial perturbations and intrinsic model errors. Hydra employs an Action-Critique Loop, where it retrieves and critiques visual information, leveraging Chain-of-Thought (CoT) and In-Context Learning (ICL) techniques to refine outputs dynamically. Unlike static post-hoc correction methods, Hydra adapts to both adversarial manipulations and intrinsic model errors, making it robust to malicious perturbations and hallucination-related inaccuracies. We evaluate Hydra on four VLMs, three hallucination benchmarks, two adversarial attack strategies, and two adversarial defense methods, assessing performance on both clean and adversarial inputs. Results show that Hydra surpasses plug-in VLMs and state-of-the-art (SOTA) dehallucination methods, even without explicit adversarial defenses, demonstrating enhanced robustness and factual consistency. By bridging adversarial resistance and hallucination mitigation, Hydra provides a scalable, training-free solution for improving the reliability of VLMs in real-world applications.

摘要: 为了开发值得信赖的视觉语言模型（VLM），解决对抗鲁棒性和幻觉缓解问题至关重要，这两者都会影响国防和医疗保健等高风险应用中的事实准确性。现有的方法主要集中在对抗防御或幻觉事后纠正上，在统一的稳健性策略方面留下了空白。我们引入了\textBF{Hydra}，这是一个自适应代理框架，通过迭代推理、结构化批评和跨模型验证来增强插件VLM，提高对对抗性扰动和固有模型错误的弹性。Hydra采用了一个“批评循环”，检索和批评视觉信息，利用思想链（CoT）和上下文学习（ICL）技术来动态细化输出。与静态事后校正方法不同，Hydra适应对抗性操作和固有模型误差，使其对恶意扰动和幻觉相关的不准确性具有鲁棒性。我们在四个VLM，三个幻觉基准，两个对抗性攻击策略和两个对抗性防御方法上评估Hydra，评估清洁和对抗性输入的性能。结果表明，即使没有明确的对抗性防御，Hydra也超过了插件VLM和最先进的（SOTA）去幻觉方法，表现出增强的鲁棒性和事实一致性。通过桥接对抗性抵抗和幻觉缓解，Hydra提供了一种可扩展的、无需培训的解决方案，用于提高VLM在现实应用中的可靠性。



## **33. WeiDetect: Weibull Distribution-Based Defense against Poisoning Attacks in Federated Learning for Network Intrusion Detection Systems**

WeiDetect：网络入侵检测系统联邦学习中基于威布尔分布的中毒攻击防御 cs.CR

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2504.04367v2) [paper-pdf](http://arxiv.org/pdf/2504.04367v2)

**Authors**: Sameera K. M., Vinod P., Anderson Rocha, Rafidha Rehiman K. A., Mauro Conti

**Abstract**: In the era of data expansion, ensuring data privacy has become increasingly critical, posing significant challenges to traditional AI-based applications. In addition, the increasing adoption of IoT devices has introduced significant cybersecurity challenges, making traditional Network Intrusion Detection Systems (NIDS) less effective against evolving threats, and privacy concerns and regulatory restrictions limit their deployment. Federated Learning (FL) has emerged as a promising solution, allowing decentralized model training while maintaining data privacy to solve these issues. However, despite implementing privacy-preserving technologies, FL systems remain vulnerable to adversarial attacks. Furthermore, data distribution among clients is not heterogeneous in the FL scenario. We propose WeiDetect, a two-phase, server-side defense mechanism for FL-based NIDS that detects malicious participants to address these challenges. In the first phase, local models are evaluated using a validation dataset to generate validation scores. These scores are then analyzed using a Weibull distribution, identifying and removing malicious models. We conducted experiments to evaluate the effectiveness of our approach in diverse attack settings. Our evaluation included two popular datasets, CIC-Darknet2020 and CSE-CIC-IDS2018, tested under non-IID data distributions. Our findings highlight that WeiDetect outperforms state-of-the-art defense approaches, improving higher target class recall up to 70% and enhancing the global model's F1 score by 1% to 14%.

摘要: 在数据扩展时代，确保数据隐私变得越来越重要，这对传统的基于人工智能的应用构成了重大挑战。此外，物联网设备的日益普及带来了重大的网络安全挑战，使传统的网络入侵检测系统（NIDS）对不断变化的威胁的有效性减弱，隐私问题和监管限制也限制了其部署。联邦学习（FL）已成为一种有前途的解决方案，它允许去中心化模型训练，同时维护数据隐私来解决这些问题。然而，尽管实施了隐私保护技术，FL系统仍然容易受到对抗攻击。此外，在FL场景中，客户端之间的数据分布并不是异类。我们提出了WeiDetect，这是一种用于基于FL的NIDS的两阶段服务器端防御机制，可以检测恶意参与者以解决这些挑战。在第一阶段，使用验证数据集评估本地模型以生成验证分数。然后使用威布尔分布分析这些分数，识别和删除恶意模型。我们进行了实验来评估我们的方法在不同攻击环境中的有效性。我们的评估包括两个流行的数据集：CIC-Darknint 2020和CSE-CIC-IDS 2018，它们在非IID数据分布下进行了测试。我们的研究结果强调，WeiDetect优于最先进的防御方法，将更高的目标类别召回率提高高达70%，并将全球车型的F1得分提高1%至14%。



## **34. Reason2Attack: Jailbreaking Text-to-Image Models via LLM Reasoning**

Reason2Attack：通过LLM推理破解文本到图像模型 cs.CR

This paper includes model-generated content that may contain  offensive or distressing material

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2503.17987v2) [paper-pdf](http://arxiv.org/pdf/2503.17987v2)

**Authors**: Chenyu Zhang, Lanjun Wang, Yiwen Ma, Wenhui Li, An-An Liu

**Abstract**: Text-to-Image(T2I) models typically deploy safety filters to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attack methods manually design prompts for the LLM to generate adversarial prompts, which effectively bypass safety filters while producing sensitive images, exposing safety vulnerabilities of T2I models. However, due to the LLM's limited understanding of the T2I model and its safety filters, existing methods require numerous queries to achieve a successful attack, limiting their practical applicability. To address this issue, we propose Reason2Attack(R2A), which aims to enhance the LLM's reasoning capabilities in generating adversarial prompts by incorporating the jailbreaking attack into the post-training process of the LLM. Specifically, we first propose a CoT example synthesis pipeline based on Frame Semantics, which generates adversarial prompts by identifying related terms and corresponding context illustrations. Using CoT examples generated by the pipeline, we fine-tune the LLM to understand the reasoning path and format the output structure. Subsequently, we incorporate the jailbreaking attack task into the reinforcement learning process of the LLM and design an attack process reward that considers prompt length, prompt stealthiness, and prompt effectiveness, aiming to further enhance reasoning accuracy. Extensive experiments on various T2I models show that R2A achieves a better attack success ratio while requiring fewer queries than baselines. Moreover, our adversarial prompts demonstrate strong attack transferability across both open-source and commercial T2I models.

摘要: 文本到图像（T2I）模型通常部署安全过滤器以防止生成敏感图像。不幸的是，最近的越狱攻击方法手动设计LLM的提示以生成对抗性提示，这有效地绕过了安全过滤器，同时生成敏感图像，暴露了T2I模型的安全漏洞。然而，由于LLM对T2I模型及其安全过滤器的理解有限，现有方法需要大量查询才能实现成功的攻击，从而限制了它们的实用性。为了解决这个问题，我们提出了Reason2Attack（R2A），旨在通过将越狱攻击纳入LLM的后训练过程来增强LLM在生成对抗性提示时的推理能力。具体来说，我们首先提出了一个基于框架语义的CoT示例合成管道，该管道通过识别相关术语和相应的上下文插图来生成对抗性提示。使用管道生成的CoT示例，我们微调LLM以了解推理路径并格式化输出结构。随后，我们将越狱攻击任务纳入LLM的强化学习过程中，并设计了考虑提示长度、提示隐蔽性和提示有效性的攻击过程奖励，旨在进一步增强推理准确性。对各种T2 I模型的广泛实验表明，R2 A实现了更好的攻击成功率，同时需要比基线更少的查询。此外，我们的对抗提示在开源和商业T2 I模型中都表现出强大的攻击可转移性。



## **35. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

Accepted to ICLR 2025. Code:  https://github.com/zhxieml/remiss-jailbreak

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2406.14393v5) [paper-pdf](http://arxiv.org/pdf/2406.14393v5)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.

摘要: 大型语言模型（LLM）的广泛采用引发了人们对其安全性和可靠性的担忧，特别是对其容易受到对抗攻击的影响。在本文中，我们提出了一种新颖的视角，将此漏洞归因于对齐过程中的奖励错误指定。当奖励函数未能准确捕捉预期行为时，就会发生这种错误规范，从而导致模型输出不一致。我们引入了一个指标ReGap来量化奖励错误指定的程度，并展示其在检测有害后门提示方面的有效性和稳健性。在这些见解的基础上，我们介绍了ReMiss，这是一个自动化红色团队系统，可以在奖励错误指定的空间中生成对抗提示。ReMiss在AdvBench基准上针对各种目标对齐的LLM实现了最先进的攻击成功率，同时保留了生成提示的人类可读性。此外，这些对开源模型的攻击证明了对GPT-4 o等闭源模型和HarmBench的分发外任务的高度可移植性。详细的分析强调了与之前的方法相比，拟议的奖励错误指定目标的独特优势，为提高LLM的安全性和稳健性提供了新的见解。



## **36. Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach**

重新思考对抗性攻击中的目标标签条件反射：2D张量引导的生成方法 cs.CV

12 pages, 4 figures

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2504.14137v1) [paper-pdf](http://arxiv.org/pdf/2504.14137v1)

**Authors**: Hangyu Liu, Bo Peng, Pengxiang Ding, Donglin Wang

**Abstract**: Compared to single-target adversarial attacks, multi-target attacks have garnered significant attention due to their ability to generate adversarial images for multiple target classes simultaneously. Existing generative approaches for multi-target attacks mainly analyze the effect of the use of target labels on noise generation from a theoretical perspective, lacking practical validation and comprehensive summarization. To address this gap, we first identify and validate that the semantic feature quality and quantity are critical factors affecting the transferability of targeted attacks: 1) Feature quality refers to the structural and detailed completeness of the implanted target features, as deficiencies may result in the loss of key discriminative information; 2) Feature quantity refers to the spatial sufficiency of the implanted target features, as inadequacy limits the victim model's attention to this feature. Based on these findings, we propose the 2D Tensor-Guided Adversarial Fusion (2D-TGAF) framework, which leverages the powerful generative capabilities of diffusion models to encode target labels into two-dimensional semantic tensors for guiding adversarial noise generation. Additionally, we design a novel masking strategy tailored for the training process, ensuring that parts of the generated noise retain complete semantic information about the target class. Extensive experiments on the standard ImageNet dataset demonstrate that 2D-TGAF consistently surpasses state-of-the-art methods in attack success rates, both on normally trained models and across various defense mechanisms.

摘要: 与单目标对抗攻击相比，多目标攻击因能够同时为多个目标类别生成对抗图像而受到了广泛关注。现有的多目标攻击生成方法主要从理论角度分析目标标签的使用对噪音产生的影响，缺乏实际验证和全面总结。为了解决这一差距，我们首先确定并验证语义特征的质量和数量是影响目标攻击可转移性的关键因素：1）特征质量是指植入目标特征的结构和细节完整性，因为缺陷可能会导致关键识别信息的丢失; 2）特征量是指植入目标特征的空间充分性，因为不充分性限制了受害者模型对该特征的注意力。基于这些发现，我们提出了2D张量引导对抗融合（2D-TGAF）框架，该框架利用扩散模型强大的生成能力将目标标签编码为二维语义张量，以引导对抗性噪音的生成。此外，我们设计了一种针对训练过程量身定制的新颖掩蔽策略，确保生成的部分噪音保留有关目标类的完整语义信息。对标准ImageNet数据集的大量实验表明，无论是在正常训练的模型上还是在各种防御机制上，2D-TGAF在攻击成功率方面始终优于最先进的方法。



## **37. Robust Decentralized Quantum Kernel Learning for Noisy and Adversarial Environment**

针对噪音和对抗环境的稳健分散量子核学习 quant-ph

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13782v1) [paper-pdf](http://arxiv.org/pdf/2504.13782v1)

**Authors**: Wenxuan Ma, Kuan-Cheng Chen, Shang Yu, Mengxiang Liu, Ruilong Deng

**Abstract**: This paper proposes a general decentralized framework for quantum kernel learning (QKL). It has robustness against quantum noise and can also be designed to defend adversarial information attacks forming a robust approach named RDQKL. We analyze the impact of noise on QKL and study the robustness of decentralized QKL to the noise. By integrating robust decentralized optimization techniques, our method is able to mitigate the impact of malicious data injections across multiple nodes. Experimental results demonstrate that our approach maintains high accuracy under noisy quantum operations and effectively counter adversarial modifications, offering a promising pathway towards the future practical, scalable and secure quantum machine learning (QML).

摘要: 本文提出了一个通用的量子核学习（QKL）的分散框架。它对量子噪声具有鲁棒性，也可以设计用于防御对抗性信息攻击，形成一种名为RDQKL的鲁棒方法。我们分析了噪声对QKL的影响，并研究了分散QKL对噪声的鲁棒性。通过集成强大的分散优化技术，我们的方法能够减轻跨多个节点的恶意数据注入的影响。实验结果表明，我们的方法在有噪音的量子操作下保持了高准确性，并有效地对抗对抗修改，为未来实用、可扩展和安全的量子机器学习（QML）提供了一条有希望的途径。



## **38. Adversarial Hubness in Multi-Modal Retrieval**

多模式检索中的对抗性积极性 cs.CR

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2412.14113v2) [paper-pdf](http://arxiv.org/pdf/2412.14113v2)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries.   In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts.   We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system implemented by Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub, generated with respect to 100 randomly selected target queries, is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries), demonstrating the strong generalization capabilities of adversarial hubs. We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.

摘要: Hubness是多维载体空间中的一种现象，其中自然分布的单个点与许多其他点异常接近。这是信息检索中一个众所周知的问题，会导致某些项意外（且错误地）看起来与许多查询相关。   在本文中，我们研究攻击者如何利用中心将多模式检索系统中的任何图像或音频输入变成对抗中心。对抗中心可用于注入通用对抗内容（例如，垃圾邮件）将根据数千个不同的查询进行检索，并对与攻击者选择的特定概念相关的查询进行有针对性的攻击。   我们提出了一种创建对抗中心的方法，并在基准多模式检索数据集和由流行的载体数据库Pinecone实现的图像到图像检索系统上评估所得中心。例如，在文本标题到图像检索中，针对100个随机选择的目标查询生成的单个对抗中心被检索为25，000个测试查询中超过21，000个的前1最相关图像（相比之下，最常见的自然中心是仅对102个查询的前1响应），这表明了对抗性中心的强大概括能力。我们还调查了减轻自然中心的技术是否是对抗性中心的有效防御，并表明它们对针对与特定概念相关的查询的中心无效。



## **39. Energy-Latency Attacks via Sponge Poisoning**

通过海绵中毒进行能量潜伏攻击 cs.CR

Paper accepted at Information Sciences journal; 20 pages Keywords:  energy-latency attacks, sponge attack, machine learning security, adversarial  machine learning

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2203.08147v5) [paper-pdf](http://arxiv.org/pdf/2203.08147v5)

**Authors**: Antonio Emanuele Cinà, Ambra Demontis, Battista Biggio, Fabio Roli, Marcello Pelillo

**Abstract**: Sponge examples are test-time inputs optimized to increase energy consumption and prediction latency of deep networks deployed on hardware accelerators. By increasing the fraction of neurons activated during classification, these attacks reduce sparsity in network activation patterns, worsening the performance of hardware accelerators. In this work, we present a novel training-time attack, named sponge poisoning, which aims to worsen energy consumption and prediction latency of neural networks on any test input without affecting classification accuracy. To stage this attack, we assume that the attacker can control only a few model updates during training -- a likely scenario, e.g., when model training is outsourced to an untrusted third party or distributed via federated learning. Our extensive experiments on image classification tasks show that sponge poisoning is effective, and that fine-tuning poisoned models to repair them poses prohibitive costs for most users, highlighting that tackling sponge poisoning remains an open issue.

摘要: Sponge示例是测试时输入，经过优化，以增加部署在硬件加速器上的深度网络的能耗和预测延迟。通过增加分类期间激活的神经元的比例，这些攻击降低了网络激活模式的稀疏性，从而恶化了硬件加速器的性能。在这项工作中，我们提出了一种新的训练时间攻击，称为海绵中毒，其目的是在不影响分类精度的情况下，恶化神经网络在任何测试输入上的能耗和预测延迟。为了进行这种攻击，我们假设攻击者在训练过程中只能控制一些模型更新--这是一种可能的情况，例如，当模型训练外包给不受信任的第三方或通过联邦学习分发时。我们对图像分类任务的广泛实验表明，海绵中毒是有效的，并且微调中毒模型以修复它们对大多数用户来说会带来高昂的成本，这凸显了解决海绵中毒仍然是一个悬而未决的问题。



## **40. Fairness and Robustness in Machine Unlearning**

机器去学习中的公平性和鲁棒性 cs.LG

5 pages

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13610v1) [paper-pdf](http://arxiv.org/pdf/2504.13610v1)

**Authors**: Khoa Tran, Simon S. Woo

**Abstract**: Machine unlearning poses the challenge of ``how to eliminate the influence of specific data from a pretrained model'' in regard to privacy concerns. While prior research on approximated unlearning has demonstrated accuracy and efficiency in time complexity, we claim that it falls short of achieving exact unlearning, and we are the first to focus on fairness and robustness in machine unlearning algorithms. Our study presents fairness Conjectures for a well-trained model, based on the variance-bias trade-off characteristic, and considers their relevance to robustness. Our Conjectures are supported by experiments conducted on the two most widely used model architectures, ResNet and ViT, demonstrating the correlation between fairness and robustness: \textit{the higher fairness-gap is, the more the model is sensitive and vulnerable}. In addition, our experiments demonstrate the vulnerability of current state-of-the-art approximated unlearning algorithms to adversarial attacks, where their unlearned models suffer a significant drop in accuracy compared to the exact-unlearned models. We claim that our fairness-gap measurement and robustness metric should be used to evaluate the unlearning algorithm. Furthermore, we demonstrate that unlearning in the intermediate and last layers is sufficient and cost-effective for time and memory complexity.

摘要: 在隐私问题方面，机器取消学习带来了“如何消除预训练模型中特定数据的影响”的挑战。虽然之前关于近似去学习的研究已经证明了时间复杂性的准确性和效率，但我们声称它未能实现精确去学习，而且我们是第一个关注机器去学习算法中的公平性和鲁棒性的人。我们的研究基于方差偏差权衡特征，提出了训练有素的模型的公平性猜想，并考虑了它们与稳健性的相关性。我们的猜想得到了在两种最广泛使用的模型架构ResNet和ViT上进行的实验的支持，证明了公平性和鲁棒性之间的相关性：\textit{公平差距越大，模型就越敏感和脆弱}。此外，我们的实验还证明了当前最先进的逼近非学习算法对对抗性攻击的脆弱性，与精确非学习模型相比，它们的非学习模型的准确性显着下降。我们声称应该使用我们的公平差距测量和稳健性指标来评估取消学习算法。此外，我们证明，对于时间和内存复杂性来说，在中间层和最后层中取消学习是足够的且具有成本效益的。



## **41. Q-FAKER: Query-free Hard Black-box Attack via Controlled Generation**

Q-FAKER：通过受控生成进行无查询硬黑匣子攻击 cs.CR

NAACL 2025 Findings

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13551v1) [paper-pdf](http://arxiv.org/pdf/2504.13551v1)

**Authors**: CheolWon Na, YunSeok Choi, Jee-Hyong Lee

**Abstract**: Many adversarial attack approaches are proposed to verify the vulnerability of language models. However, they require numerous queries and the information on the target model. Even black-box attack methods also require the target model's output information. They are not applicable in real-world scenarios, as in hard black-box settings where the target model is closed and inaccessible. Even the recently proposed hard black-box attacks still require many queries and demand extremely high costs for training adversarial generators. To address these challenges, we propose Q-faker (Query-free Hard Black-box Attacker), a novel and efficient method that generates adversarial examples without accessing the target model. To avoid accessing the target model, we use a surrogate model instead. The surrogate model generates adversarial sentences for a target-agnostic attack. During this process, we leverage controlled generation techniques. We evaluate our proposed method on eight datasets. Experimental results demonstrate our method's effectiveness including high transferability and the high quality of the generated adversarial examples, and prove its practical in hard black-box settings.

摘要: 人们提出了许多对抗攻击方法来验证语言模型的脆弱性。然而，它们需要大量查询和有关目标模型的信息。即使是黑匣子攻击方法也需要目标模型的输出信息。它们不适用于现实世界场景，例如在目标模型关闭且无法访问的硬黑匣子设置中。即使是最近提出的硬黑匣子攻击仍然需要许多查询，并且需要极高的训练对抗生成器的成本。为了应对这些挑战，我们提出了Q-faker（无查询硬黑匣子攻击者），这是一种新颖且高效的方法，可以在不访问目标模型的情况下生成对抗性示例。为了避免访问目标模型，我们使用代理模型。代理模型为目标不可知攻击生成对抗性句子。在此过程中，我们利用受控发电技术。我们在八个数据集上评估了我们提出的方法。实验结果表明，该方法的有效性，包括高的可移植性和高质量的生成的对抗性的例子，并证明其实用性在硬黑盒设置。



## **42. Few-shot Model Extraction Attacks against Sequential Recommender Systems**

针对顺序推荐系统的少样本模型抽取攻击 cs.LG

It requires substantial modifications.The symbols in the mathematical  formulas are not explained in detail

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2411.11677v2) [paper-pdf](http://arxiv.org/pdf/2411.11677v2)

**Authors**: Hui Zhang, Fu Liu

**Abstract**: Among adversarial attacks against sequential recommender systems, model extraction attacks represent a method to attack sequential recommendation models without prior knowledge. Existing research has primarily concentrated on the adversary's execution of black-box attacks through data-free model extraction. However, a significant gap remains in the literature concerning the development of surrogate models by adversaries with access to few-shot raw data (10\% even less). That is, the challenge of how to construct a surrogate model with high functional similarity within the context of few-shot data scenarios remains an issue that requires resolution.This study addresses this gap by introducing a novel few-shot model extraction framework against sequential recommenders, which is designed to construct a superior surrogate model with the utilization of few-shot data. The proposed few-shot model extraction framework is comprised of two components: an autoregressive augmentation generation strategy and a bidirectional repair loss-facilitated model distillation procedure. Specifically, to generate synthetic data that closely approximate the distribution of raw data, autoregressive augmentation generation strategy integrates a probabilistic interaction sampler to extract inherent dependencies and a synthesis determinant signal module to characterize user behavioral patterns. Subsequently, bidirectional repair loss, which target the discrepancies between the recommendation lists, is designed as auxiliary loss to rectify erroneous predictions from surrogate models, transferring knowledge from the victim model to the surrogate model effectively. Experiments on three datasets show that the proposed few-shot model extraction framework yields superior surrogate models.

摘要: 在针对顺序推荐系统的对抗性攻击中，模型提取攻击代表了一种在没有先验知识的情况下攻击顺序推荐模型的方法。现有的研究主要集中在对手通过无数据模型提取执行黑匣子攻击。然而，关于对手能够获得少量原始数据（10%甚至更少）开发代理模型的文献中仍然存在显着差距。也就是说，如何在少镜头数据场景的背景下构建具有高功能相似性的代理模型的挑战仍然是一个需要解决的问题。本研究通过引入一种针对顺序排序器的新型少镜头模型提取框架来解决这一差距，该框架旨在利用少镜头数据构建更好的代理模型。提出的少镜头模型提取框架由两个部分组成：自回归增强生成策略和双向修复损失促进模型蒸馏过程。具体来说，为了生成非常接近原始数据分布的合成数据，自回归增强生成策略集成了用于提取固有依赖关系的概率交互采样器和用于描述用户行为模式的合成决定因素信号模块。随后，针对推荐列表之间的差异，将双向修复损失设计为辅助损失，以纠正代理模型的错误预测，有效地将知识从受害者模型转移到代理模型。对三个数据集的实验表明，提出的少镜头模型提取框架可以产生更好的代理模型。



## **43. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

WWW'25 research track accepted

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2406.11260v3) [paper-pdf](http://arxiv.org/pdf/2406.11260v3)

**Authors**: Sungwon Park, Sungwon Han, Xing Xie, Jae-Gil Lee, Meeyoung Cha

**Abstract**: The spread of fake news harms individuals and presents a critical social challenge that must be addressed. Although numerous algorithmic and insightful features have been developed to detect fake news, many of these features can be manipulated with style-conversion attacks, especially with the emergence of advanced language models, making it more difficult to differentiate from genuine news. This study proposes adversarial style augmentation, AdStyle, designed to train a fake news detector that remains robust against various style-conversion attacks. The primary mechanism involves the strategic use of LLMs to automatically generate a diverse and coherent array of style-conversion attack prompts, enhancing the generation of particularly challenging prompts for the detector. Experiments indicate that our augmentation strategy significantly improves robustness and detection performance when evaluated on fake news benchmark datasets.

摘要: 假新闻的传播伤害了个人，并提出了必须解决的严重社会挑战。尽管已经开发了许多算法和有洞察力的功能来检测假新闻，但其中许多功能都可以通过风格转换攻击来操纵，特别是随着高级语言模型的出现，使其更难与真实新闻区分开来。这项研究提出了对抗性风格增强AdStyle，旨在训练假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。主要机制涉及战略性地使用LLM来自动生成多样化且连贯的风格转换攻击提示阵列，从而增强检测器特别具有挑战性的提示的生成。实验表明，当对假新闻基准数据集进行评估时，我们的增强策略显着提高了鲁棒性和检测性能。



## **44. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.05050v2) [paper-pdf](http://arxiv.org/pdf/2504.05050v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **45. EXAM: Exploiting Exclusive System-Level Cache in Apple M-Series SoCs for Enhanced Cache Occupancy Attacks**

EXAM：利用Apple M系列SOC中的独占系统级缓存进行增强型缓存占用攻击 cs.CR

Accepted to ACM ASIA CCS 2025

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13385v1) [paper-pdf](http://arxiv.org/pdf/2504.13385v1)

**Authors**: Tianhong Xu, Aidong Adam Ding, Yunsi Fei

**Abstract**: Cache occupancy attacks exploit the shared nature of cache hierarchies to infer a victim's activities by monitoring overall cache usage, unlike access-driven cache attacks that focus on specific cache lines or sets. There exists some prior work that target the last-level cache (LLC) of Intel processors, which is inclusive of higher-level caches, and L2 caches of ARM systems. In this paper, we target the System-Level Cache (SLC) of Apple M-series SoCs, which is exclusive to higher-level CPU caches. We address the challenges of the exclusiveness and propose a suite of SLC-cache occupancy attacks, the first of its kind, where an adversary can monitor GPU and other CPU cluster activities from their own CPU cluster. We first discover the structure of SLC in Apple M1 SOC and various policies pertaining to access and sharing through reverse engineering. We propose two attacks against websites. One is a coarse-grained fingerprinting attack, recognizing which website is accessed based on their different GPU memory access patterns monitored through the SLC occupancy channel. The other attack is a fine-grained pixel stealing attack, which precisely monitors the GPU memory usage for rendering different pixels, through the SLC occupancy channel. Third, we introduce a novel screen capturing attack which works beyond webpages, with the monitoring granularity of 57 rows of pixels (there are 1600 rows for the screen). This significantly expands the attack surface, allowing the adversary to retrieve any screen display, posing a substantial new threat to system security. Our findings reveal critical vulnerabilities in Apple's M-series SoCs and emphasize the urgent need for effective countermeasures against cache occupancy attacks in heterogeneous computing environments.

摘要: 缓存占用率攻击利用缓存层次结构的共享性质，通过监控总体缓存使用情况来推断受害者的活动，这与专注于特定缓存行或集的访问驱动缓存攻击不同。以前有一些针对英特尔处理器的最后一级缓存（LLC）的工作，其中包括更高级的缓存和ARM系统的L2缓存。在本文中，我们的目标是Apple M系列SOC的系统级缓存（SLC），该缓存专为更高级的中央处理器缓存。我们解决了独占性的挑战，并提出了一套SLC缓存占用率攻击，这是此类攻击中的第一个，对手可以通过其自己的中央处理器集群监控图形处理器和其他中央处理器集群活动。我们首先发现Apple M1 SOC中SLC的结构以及与通过反向工程访问和共享相关的各种政策。我们提出了两种针对网站的攻击。一种是粗粒度指纹攻击，根据通过SLC占用通道监控的不同图形处理器内存访问模式来识别访问哪个网站。另一种攻击是细粒度像素窃取攻击，该攻击通过SLC占用通道精确监控渲染不同像素的图形处理器内存使用情况。第三，我们引入了一种新颖的屏幕捕获攻击，其适用于网页之外，监控粒度为57行像素（屏幕有1600行）。这显着扩大了攻击面，允许对手检索任何屏幕显示，对系统安全构成了重大的新威胁。我们的调查结果揭示了苹果M系列SOC中的关键漏洞，并强调迫切需要针对异类计算环境中的缓存占用攻击采取有效的应对措施。



## **46. DYNAMITE: Dynamic Defense Selection for Enhancing Machine Learning-based Intrusion Detection Against Adversarial Attacks**

CLARITE：动态防御选择，以增强基于机器学习的入侵检测对抗性攻击 cs.CR

Accepted by the IEEE/ACM Workshop on the Internet of Safe Things  (SafeThings 2025)

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13301v1) [paper-pdf](http://arxiv.org/pdf/2504.13301v1)

**Authors**: Jing Chen, Onat Gungor, Zhengli Shang, Elvin Li, Tajana Rosing

**Abstract**: The rapid proliferation of the Internet of Things (IoT) has introduced substantial security vulnerabilities, highlighting the need for robust Intrusion Detection Systems (IDS). Machine learning-based intrusion detection systems (ML-IDS) have significantly improved threat detection capabilities; however, they remain highly susceptible to adversarial attacks. While numerous defense mechanisms have been proposed to enhance ML-IDS resilience, a systematic approach for selecting the most effective defense against a specific adversarial attack remains absent. To address this challenge, we propose Dynamite, a dynamic defense selection framework that enhances ML-IDS by intelligently identifying and deploying the most suitable defense using a machine learning-driven selection mechanism. Our results demonstrate that Dynamite achieves a 96.2% reduction in computational time compared to the Oracle, significantly decreasing computational overhead while preserving strong prediction performance. Dynamite also demonstrates an average F1-score improvement of 76.7% over random defense and 65.8% over the best static state-of-the-art defense.

摘要: 物联网（IOT）的迅速普及引入了大量的安全漏洞，凸显了对强大的入侵检测系统（IDS）的需求。基于机器学习的入侵检测系统（ML-IDS）显着提高了威胁检测能力;然而，它们仍然极易受到对抗性攻击。虽然已经提出了多种防御机制来增强ML-IDS的弹性，但仍然缺乏一种系统性的方法来选择针对特定对抗攻击的最有效防御。为了应对这一挑战，我们提出了Dynamite，这是一个动态防御选择框架，它通过使用机器学习驱动的选择机制智能识别和部署最合适的防御来增强ML-IDS。我们的结果表明，与Oracle相比，Dynamite的计算时间减少了96.2%，显着降低了计算负担，同时保持了强劲的预测性能。Dynamite还表明，F1平均得分比随机防守提高了76.7%，比最佳静态最先进防御提高了65.8%。



## **47. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

通过简单的自适应攻击越狱领先的安全一致LLM cs.CR

Accepted at ICLR 2025. Updates in the v3: GPT-4o and Claude 3.5  Sonnet results, improved writing. Updates in the v2: more models (Llama3,  Phi-3, Nemotron-4-340B), jailbreak artifacts for all attacks are available,  evaluation with different judges (Llama-3-70B and Llama Guard 2), more  experiments (convergence plots, ablation on the suffix length for random  search), examples of jailbroken generation

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2404.02151v4) [paper-pdf](http://arxiv.org/pdf/2404.02151v4)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4o, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.

摘要: 我们表明，即使是最新的安全一致的LLM也无法抵御简单的自适应越狱攻击。首先，我们演示了如何成功利用对logprob的访问进行越狱：我们最初设计一个对抗提示模板（有时适合目标LLM），然后对后缀应用随机搜索以最大化目标logprob（例如，令牌“Sure”），可能会多次重新启动。通过这种方式，我们在Vicuna-13 B、Mistral-7 B、Phi-3-Mini、Nemotron-4- 340 B、Llama-2-Chat-7 B/13 B/70 B、Llama-3-Direct-8B、Gemma-7 B、GPT-3.5、GPT-4 o和来自HarmBench的R2 D2上实现了100%的攻击成功率，该公司接受了针对GCG攻击的对抗训练。我们还展示了如何通过传输或预填充攻击以100%的成功率越狱所有Claude模型（不会暴露logprobs）。此外，我们还展示了如何对一组受限制的令牌使用随机搜索来在中毒模型中寻找特洛伊木马字符串--这项任务与越狱有许多相似之处--该算法使我们在SaTML ' 24特洛伊木马检测竞赛中获得第一名。这些攻击背后的共同主题是自适应性至关重要：不同的模型容易受到不同提示模板的影响（例如，R2 D2对上下文学习提示非常敏感），一些模型根据其API存在独特的漏洞（例如，为Claude预填充），在某些设置中，基于先验知识限制令牌搜索空间至关重要（例如，用于木马检测）。出于重现性的目的，我们在https://github.com/tml-epfl/llm-adaptive-attacks上提供了JailbreakBench格式的代码、日志和越狱工件。



## **48. Chypnosis: Stealthy Secret Extraction using Undervolting-based Static Side-channel Attacks**

Chypnosis：使用基于欠压的静态侧通道攻击进行秘密提取 cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.11633v2) [paper-pdf](http://arxiv.org/pdf/2504.11633v2)

**Authors**: Kyle Mitard, Saleh Khalaj Monfared, Fatemeh Khojasteh Dana, Shahin Tajik

**Abstract**: There is a growing class of static physical side-channel attacks that allow adversaries to extract secrets by probing the persistent state of a circuit. Techniques such as laser logic state imaging (LLSI), impedance analysis (IA), and static power analysis fall into this category. These attacks require that the targeted data remain constant for a specific duration, which often necessitates halting the circuit's clock. Some methods additionally rely on modulating the chip's supply voltage to probe the circuit. However, tampering with the clock or voltage is typically assumed to be detectable, as secure chips often deploy sensors that erase sensitive data upon detecting such anomalies. Furthermore, many secure devices use internal clock sources, making external clock control infeasible. In this work, we introduce a novel class of static side-channel attacks, called Chypnosis, that enables adversaries to freeze a chip's internal clock by inducing a hibernation state via rapid undervolting, and then extracting secrets using static side-channels. We demonstrate that, by rapidly dropping a chip's voltage below the standard nominal levels, the attacker can bypass the clock and voltage sensors and put the chip in a so-called brownout condition, in which the chip's transistors stop switching, but volatile memories (e.g., Flip-flops and SRAMs) still retain their data. We test our attack on AMD FPGAs by putting them into hibernation. We show that not only are all clock sources deactivated, but various clock and voltage sensors also fail to detect the tamper event. Afterward, we present the successful recovery of secret bits from a hibernated chip using two static attacks, namely, LLSI and IA. Finally, we discuss potential countermeasures which could be integrated into future designs.

摘要: 越来越多的静态物理侧通道攻击允许对手通过探测电路的持续状态来提取秘密。激光逻辑状态成像（LLSI）、阻抗分析（IA）和静态功率分析等技术都属于这一类。这些攻击要求目标数据在特定的持续时间内保持不变，这通常需要停止电路的时钟。有些方法还依赖于调制芯片的电源电压来探测电路。然而，对时钟或电压的篡改通常被认为是可检测的，因为安全芯片通常部署传感器，在检测到此类异常时擦除敏感数据。此外，许多安全设备使用内部时钟源，使得外部时钟控制不可行。在这项工作中，我们引入了一类新型的静态侧通道攻击，称为Chypnosis，它使对手能够通过快速欠电压诱导休眠状态来冻结芯片的内部时钟，然后使用静态侧通道提取秘密。我们证明，通过将芯片的电压快速降低到标准名义水平以下，攻击者可以绕过时钟和电压传感器，并将芯片置于所谓的停电条件，其中芯片的晶体管停止切换，但易失性存储器（例如，人字拖和RAM）仍然保留其数据。我们通过将AMD VGA置于休眠状态来测试对它们的攻击。我们表明，不仅所有时钟源都被停用，而且各种时钟和电压传感器也无法检测到篡改事件。随后，我们展示了使用两种静态攻击（LLSI和IA）从休眠芯片中成功恢复秘密位的方法。最后，我们讨论了可以整合到未来设计中的潜在对策。



## **49. Strategic Planning of Stealthy Backdoor Attacks in Markov Decision Processes**

马尔科夫决策过程中隐形后门攻击的战略规划 eess.SY

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13276v1) [paper-pdf](http://arxiv.org/pdf/2504.13276v1)

**Authors**: Xinyi Wei, Shuo Han, Ahmed H. Hemida, Charles A. Kamhoua, Jie Fu

**Abstract**: This paper investigates backdoor attack planning in stochastic control systems modeled as Markov Decision Processes (MDPs). In a backdoor attack, the adversary provides a control policy that behaves well in the original MDP to pass the testing phase. However, when such a policy is deployed with a trigger policy, which perturbs the system dynamics at runtime, it optimizes the attacker's objective instead. To solve jointly the control policy and its trigger, we formulate the attack planning problem as a constrained optimal planning problem in an MDP with augmented state space, with the objective to maximize the attacker's total rewards in the system with an activated trigger, subject to the constraint that the control policy is near optimal in the original MDP. We then introduce a gradient-based optimization method to solve the optimal backdoor attack policy as a pair of coordinated control and trigger policies. Experimental results from a case study validate the effectiveness of our approach in achieving stealthy backdoor attacks.

摘要: 研究了随机控制系统中的后门攻击规划问题。在后门攻击中，对手提供了一个在原始MDP中表现良好的控制策略，以通过测试阶段。然而，当这样的策略与在运行时扰乱系统动态的触发策略一起部署时，它反而优化了攻击者的目标。为了共同解决的控制策略和它的触发器，我们制定的攻击规划问题作为一个有约束的最优规划问题的MDP与增广的状态空间，目标是最大限度地提高攻击者的总回报在系统中激活触发器，受约束的控制策略是接近最优的原始MDP。然后，我们引入一种基于梯度的优化方法来解决最优后门攻击策略，作为一对协调的控制和触发策略。案例研究的实验结果验证了我们的方法在实现隐形后门攻击方面的有效性。



## **50. Does Refusal Training in LLMs Generalize to the Past Tense?**

LLM中的拒绝培训是否适用于过去时态？ cs.CL

Accepted at ICLR 2025. Updates in v2 and v3: added GPT-4o, Claude 3.5  Sonnet, o1-mini, and o1-preview results. Code and jailbreak artifacts:  https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2407.11969v4) [paper-pdf](http://arxiv.org/pdf/2407.11969v4)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, o1-mini, o1-preview, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.

摘要: 拒绝培训被广泛用于防止LLM产生有害、不良或非法的输出。我们揭示了当前拒绝训练方法中一个奇怪的概括差距：简单地用过去时重新表达有害的请求（例如，“如何制作燃烧弹？”到“人们是如何制作燃烧弹的？”）通常足以越狱许多最先进的法学硕士。我们在Llama-3 8B、Claude-3.5十四行诗、GPT-3.5 Turbo、Gemma-2 9 B、Phi-3-Mini、GPT-4 o mini、GPT-4 o mini、o 1-mini、o 1-预览和R2D2模型上系统地评估了该方法，使用GPT-3.5 Turbo作为重新制定模型。例如，这种对GPT-4 o的简单攻击的成功率从使用直接请求的1%增加到使用来自JailbreakBench的有害请求的20次过去时重新表述尝试（以GPT-4作为越狱法官）的88%。有趣的是，我们还发现未来时的重新表述效果不太好，这表明拒绝护栏往往会考虑过去的历史问题而不是假设的未来问题。此外，我们的微调GPT-3.5涡轮实验表明，防御过去的改写是可行的，过去时态的例子显式地包括在微调数据。总的来说，我们的研究结果强调了广泛使用的对齐技术-如SFT，RLHF和对抗训练-用于对齐所研究的模型可能是脆弱的，并且并不总是按预期进行推广。我们在https://github.com/tml-epfl/llm-past-tense上提供代码和越狱工件。



