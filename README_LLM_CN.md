# Latest Large Language Model Attack Papers
**update at 2025-05-24 16:11:32**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models**

看不见的警告，可见的威胁：大型语言模型的外部资源中的恶意字体注入 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16957v1) [paper-pdf](http://arxiv.org/pdf/2505.16957v1)

**Authors**: Junjie Xiong, Changjia Zhu, Shuhang Lin, Chong Zhang, Yongfeng Zhang, Yao Liu, Lingyao Li

**Abstract**: Large Language Models (LLMs) are increasingly equipped with capabilities of real-time web search and integrated with protocols like Model Context Protocol (MCP). This extension could introduce new security vulnerabilities. We present a systematic investigation of LLM vulnerabilities to hidden adversarial prompts through malicious font injection in external resources like webpages, where attackers manipulate code-to-glyph mapping to inject deceptive content which are invisible to users. We evaluate two critical attack scenarios: (1) "malicious content relay" and (2) "sensitive data leakage" through MCP-enabled tools. Our experiments reveal that indirect prompts with injected malicious font can bypass LLM safety mechanisms through external resources, achieving varying success rates based on data sensitivity and prompt design. Our research underscores the urgent need for enhanced security measures in LLM deployments when processing external content.

摘要: 大型语言模型（LLM）越来越多地配备实时网络搜索功能，并与模型上下文协议（HCP）等协议集成。此扩展可能会引入新的安全漏洞。我们对通过在网页等外部资源中恶意字体注入来隐藏对抗提示的LLM漏洞进行了系统性调查，其中攻击者操纵代码到收件箱的映射来注入用户不可见的欺骗性内容。我们评估了两种关键攻击场景：（1）“恶意内容中继”和（2）通过支持MVP的工具“敏感数据泄露”。我们的实验表明，注入恶意字体的间接提示可以通过外部资源绕过LLM安全机制，根据数据敏感性和提示设计实现不同的成功率。我们的研究强调了处理外部内容时LLM部署中迫切需要增强的安全措施。



## **2. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

MixAT：结合LLM的连续和离散对抗训练 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16947v1) [paper-pdf](http://arxiv.org/pdf/2505.16947v1)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.

摘要: 尽管最近在大型语言模型（LLM）的安全性和一致性方面做出了努力，但当前对前沿LLM的对抗性攻击仍然能够持续地迫使有害的世代。尽管对抗训练已得到广泛研究，并被证明可以显着提高传统机器学习模型的鲁棒性，但其在LLM背景下的优点和缺点却知之甚少。具体来说，虽然现有的离散对抗攻击可以有效地产生有害内容，但用具体的对抗提示训练LLM通常计算成本高昂，导致依赖于持续的放松。由于这些松弛不对应于离散输入令牌，因此此类潜在训练方法通常使模型容易受到一系列不同的离散攻击。在这项工作中，我们的目标是通过引入MixAT来弥合这一差距，MixAT是一种新颖的方法，在训练期间结合了更强的离散攻击和更快的连续攻击。我们对MixAT进行了广泛的最先进攻击，提出了至少一次攻击成功率（ALO-ASB）指标来捕捉模型的最坏情况漏洞。我们表明，与之前的防御（ALO-ASB> 50%）相比，MixAT实现了更好的鲁棒性（ALO-ASB < 20%），同时保持与基于连续松弛的方法相当的运行时间。我们进一步分析了现实部署环境中的MixAT，探索聊天模板、量化、低等级适配器和温度如何影响对抗训练和评估，从而揭示了当前方法中的其他盲点。我们的结果表明，MixAT的离散-连续防御以最小的计算负担提供了原则性且卓越的鲁棒性-准确性权衡，凸显了其构建更安全的LLM的承诺。我们在https://github.com/insait-institute/MixAT上提供我们的代码和模型。



## **3. Backdoor Cleaning without External Guidance in MLLM Fine-tuning**

MLLM微调中未经外部指导的后门清理 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16916v1) [paper-pdf](http://arxiv.org/pdf/2505.16916v1)

**Authors**: Xuankun Rong, Wenke Huang, Jian Liang, Jinhe Bi, Xun Xiao, Yiming Li, Bo Du, Mang Ye

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed in fine-tuning-as-a-service (FTaaS) settings, where user-submitted datasets adapt general-purpose models to downstream tasks. This flexibility, however, introduces serious security risks, as malicious fine-tuning can implant backdoors into MLLMs with minimal effort. In this paper, we observe that backdoor triggers systematically disrupt cross-modal processing by causing abnormal attention concentration on non-semantic regions--a phenomenon we term attention collapse. Based on this insight, we propose Believe Your Eyes (BYE), a data filtering framework that leverages attention entropy patterns as self-supervised signals to identify and filter backdoor samples. BYE operates via a three-stage pipeline: (1) extracting attention maps using the fine-tuned model, (2) computing entropy scores and profiling sensitive layers via bimodal separation, and (3) performing unsupervised clustering to remove suspicious samples. Unlike prior defenses, BYE equires no clean supervision, auxiliary labels, or model modifications. Extensive experiments across various datasets, models, and diverse trigger types validate BYE's effectiveness: it achieves near-zero attack success rates while maintaining clean-task performance, offering a robust and generalizable solution against backdoor threats in MLLMs.

摘要: 多模式大型语言模型（MLLM）越来越多地部署在微调即服务（FTSaaS）设置中，其中用户提交的数据集将通用模型适应下游任务。然而，这种灵活性会带来严重的安全风险，因为恶意微调可以以最少的努力将后门植入MLLM中。在本文中，我们观察到后门触发器通过导致非语义区域的异常注意集中来系统性地扰乱跨模式处理--我们将这种现象称为注意力崩溃。基于这一见解，我们提出了相信你的眼睛（BYE），这是一种数据过滤框架，利用注意力熵模式作为自我监督信号来识别和过滤后门样本。BYE通过三阶段流水线运行：（1）使用微调模型提取注意力图，（2）通过双峰分离计算熵分数并分析敏感层，以及（3）执行无监督集群以删除可疑样本。与之前的防御不同，BYE不提供干净的监督、辅助标签或型号修改。跨各种数据集、模型和不同触发类型的广泛实验验证了BYE的有效性：它实现了接近零的攻击成功率，同时保持干净任务性能，提供了针对MLLM中后门威胁的强大且可推广的解决方案。



## **4. CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework**

CAIN：通过两阶段恶意系统提示生成和精炼框架劫持LLM与人类对话 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16888v1) [paper-pdf](http://arxiv.org/pdf/2505.16888v1)

**Authors**: Viet Pham, Thai Le

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.

摘要: 大型语言模型（LLM）先进了许多应用程序，但也容易受到对抗攻击。在这项工作中，我们引入了一种新颖的安全威胁：通过操纵LLM的系统提示来劫持人工智能与人类的对话，以仅对特定目标问题（例如，“我应该投票给谁美国总统？”，“新冠疫苗安全吗？”），同时对他人表现友善。这种攻击是有害的，因为它可以使恶意行为者通过在线传播有害但看起来友善的系统提示来进行大规模信息操纵。为了演示此类攻击，我们开发了CAIN，这是一种算法，可以在黑匣子设置中或无需访问LLM参数的情况下自动策划此类有害系统提示特定目标问题。在开源和商业LLM上进行评估，CAIN表现出显着的对抗影响。在无针对性攻击或迫使LLM输出错误答案中，CAIN对目标问题实现了高达40%的F1降级，同时对良性输入保持高准确性。对于有针对性的攻击或迫使LLM输出特定的有害答案，CAIN在这些有针对性的回答上获得了超过70%的F1分数，而对良性问题的影响最小。我们的结果凸显了对增强稳健性措施的迫切需要，以保障LLM在现实世界应用中的完整性和安全性。所有源代码都将公开。



## **5. Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback**

Safe RLHF-V：基于多模态人类反馈的安全强化学习 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2503.17682v2) [paper-pdf](http://arxiv.org/pdf/2503.17682v2)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Conghui Zhang, Han Zhu, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Yida Tang, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are essential for building general-purpose AI assistants; however, they pose increasing safety risks. How can we ensure safety alignment of MLLMs to prevent undesired behaviors? Going further, it is critical to explore how to fine-tune MLLMs to preserve capabilities while meeting safety constraints. Fundamentally, this challenge can be formulated as a min-max optimization problem. However, existing datasets have not yet disentangled single preference signals into explicit safety constraints, hindering systematic investigation in this direction. Moreover, it remains an open question whether such constraints can be effectively incorporated into the optimization process for multi-modal models. In this work, we present the first exploration of the Safe RLHF-V -- the first multimodal safety alignment framework. The framework consists of: $\mathbf{(I)}$ BeaverTails-V, the first open-source dataset featuring dual preference annotations for helpfulness and safety, supplemented with multi-level safety labels (minor, moderate, severe); $\mathbf{(II)}$ Beaver-Guard-V, a multi-level guardrail system to proactively defend against unsafe queries and adversarial attacks. Applying the guard model over five rounds of filtering and regeneration significantly enhances the precursor model's overall safety by an average of 40.9%. $\mathbf{(III)}$ Based on dual preference, we initiate the first exploration of multi-modal safety alignment within a constrained optimization. Experimental results demonstrate that Safe RLHF effectively improves both model helpfulness and safety. Specifically, Safe RLHF-V enhances model safety by 34.2% and helpfulness by 34.3%.

摘要: 多模式大型语言模型（MLLM）对于构建通用人工智能助手至关重要;然而，它们带来了越来越大的安全风险。我们如何确保MLLM的安全一致以防止不良行为？进一步说，探索如何微调MLLM以在满足安全限制的同时保留功能至关重要。从根本上讲，这个挑战可以被描述为一个最小-最大优化问题。然而，现有的数据集尚未将单一偏好信号分解为明确的安全约束，从而阻碍了这方面的系统性研究。此外，这些约束是否可以有效地纳入多模式模型的优化过程仍然是一个悬而未决的问题。在这项工作中，我们首次探索Safe RLHF-V --第一个多模式安全对齐框架。该框架包括：$\mathBF{（I）}$ BeaverTails-V，第一个开源数据集，具有帮助性和安全性的双重偏好注释，并辅之以多级别安全标签（轻微、中度、严重）; $\mathBF{（II）}$ Beaver-Guard-V，一个多级别护栏系统，用于主动防御不安全的查询和对抗性攻击。经过五轮过滤和再生应用防护模型，前体模型的整体安全性平均显着提高了40.9%。$\mathBF{（III）}$基于双重偏好，我们在约束优化中启动了多模式安全对齐的首次探索。实验结果表明，Safe RL HF有效提高了模型的帮助性和安全性。具体而言，Safe RLHF-V将模型的安全性提高了34.2%，帮助性提高了34.3%。



## **6. Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability**

意外失调：微调语言模型会引发意外漏洞 cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16789v1) [paper-pdf](http://arxiv.org/pdf/2505.16789v1)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_misalignment.

摘要: 随着大型语言模型越来越受欢迎，它们对对抗攻击的脆弱性仍然是一个主要问题。虽然通常使用特定领域数据集的微调模型来提高模型性能，但它可能会在基础模型中引入漏洞。在这项工作中，我们调查了意外失准，即微调数据特征引起的意外漏洞。我们首先确定潜在的相关因素，如语言特征，语义相似性和毒性在我们的实验数据集。然后，我们评估这些微调模型的对抗性能，并评估数据集因素与攻击成功率的相关性。最后，我们探索了潜在的因果关系，为对抗性防御策略提供了新的见解，并强调了数据集设计在保持模型对齐方面的关键作用。我们的代码可在https://github.com/psyonp/accidental_misalignment上获取。



## **7. When Safety Detectors Aren't Enough: A Stealthy and Effective Jailbreak Attack on LLMs via Steganographic Techniques**

当安全检测器还不够时：通过隐写技术对LLM进行秘密有效的越狱攻击 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16765v1) [paper-pdf](http://arxiv.org/pdf/2505.16765v1)

**Authors**: Jianing Geng, Biao Yi, Zekun Fei, Tongxi Wu, Lihai Nie, Zheli Liu

**Abstract**: Jailbreak attacks pose a serious threat to large language models (LLMs) by bypassing built-in safety mechanisms and leading to harmful outputs. Studying these attacks is crucial for identifying vulnerabilities and improving model security. This paper presents a systematic survey of jailbreak methods from the novel perspective of stealth. We find that existing attacks struggle to simultaneously achieve toxic stealth (concealing toxic content) and linguistic stealth (maintaining linguistic naturalness). Motivated by this, we propose StegoAttack, a fully stealthy jailbreak attack that uses steganography to hide the harmful query within benign, semantically coherent text. The attack then prompts the LLM to extract the hidden query and respond in an encrypted manner. This approach effectively hides malicious intent while preserving naturalness, allowing it to evade both built-in and external safety mechanisms. We evaluate StegoAttack on four safety-aligned LLMs from major providers, benchmarking against eight state-of-the-art methods. StegoAttack achieves an average attack success rate (ASR) of 92.00%, outperforming the strongest baseline by 11.0%. Its ASR drops by less than 1% even under external detection (e.g., Llama Guard). Moreover, it attains the optimal comprehensive scores on stealth detection metrics, demonstrating both high efficacy and exceptional stealth capabilities. The code is available at https://anonymous.4open.science/r/StegoAttack-Jail66

摘要: 越狱攻击绕过内置安全机制并导致有害输出，对大型语言模型（LLM）构成严重威胁。研究这些攻击对于识别漏洞和提高模型安全性至关重要。本文从隐身的新颖角度对越狱方法进行了系统的概述。我们发现现有的攻击很难同时实现有毒隐形（隐藏有毒内容）和语言隐形（保持语言自然性）。出于此动机，我们提出了StegoAttack，这是一种完全隐蔽的越狱攻击，使用隐写术将有害查询隐藏在良性、语义连贯的文本中。然后，攻击会促使LLM提取隐藏的查询并以加密方式响应。这种方法有效地隐藏恶意意图，同时保持自然性，使其能够逃避内置和外部安全机制。我们评估StegoAttack对四个安全对齐的LLM从主要供应商，基准对八个国家的最先进的方法。StegoAttack的平均攻击成功率（ASR）为92.00%，比最强基线高出11.0%。即使在外部检测下，其ASR也下降不到1%（例如，Llama Guard）。此外，它还获得了最佳的隐身检测指标综合评分，展示了高功效和出色的隐身能力。该代码可在https://anonymous.4open.science/r/StegoAttack-Jail66上获取



## **8. BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models**

BitHydra：针对大型语言模型的位翻转推理成本攻击 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16670v1) [paper-pdf](http://arxiv.org/pdf/2505.16670v1)

**Authors**: Xiaobei Yan, Yiming Li, Zhaoxin Fan, Han Qiu, Tianwei Zhang

**Abstract**: Large language models (LLMs) have shown impressive capabilities across a wide range of applications, but their ever-increasing size and resource demands make them vulnerable to inference cost attacks, where attackers induce victim LLMs to generate the longest possible output content. In this paper, we revisit existing inference cost attacks and reveal that these methods can hardly produce large-scale malicious effects since they are self-targeting, where attackers are also the users and therefore have to execute attacks solely through the inputs, whose generated content will be charged by LLMs and can only directly influence themselves. Motivated by these findings, this paper introduces a new type of inference cost attacks (dubbed 'bit-flip inference cost attack') that target the victim model itself rather than its inputs. Specifically, we design a simple yet effective method (dubbed 'BitHydra') to effectively flip critical bits of model parameters. This process is guided by a loss function designed to suppress <EOS> token's probability with an efficient critical bit search algorithm, thus explicitly defining the attack objective and enabling effective optimization. We evaluate our method on 11 LLMs ranging from 1.5B to 14B parameters under both int8 and float16 settings. Experimental results demonstrate that with just 4 search samples and as few as 3 bit flips, BitHydra can force 100% of test prompts to reach the maximum generation length (e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its efficiency, scalability, and strong transferability across unseen inputs.

摘要: 大型语言模型（LLM）在广泛的应用程序中表现出令人印象深刻的能力，但其不断增加的规模和资源需求使它们容易受到推理成本攻击，攻击者诱导受害者LLM生成尽可能长的输出内容。在本文中，我们回顾了现有的推理成本攻击，并揭示了这些方法很难产生大规模的恶意影响，因为它们是自瞄准的，攻击者也是用户，因此必须仅通过输入来执行攻击，其生成的内容将由LLM收费并且只能直接影响自己。受这些发现的启发，本文引入了一种新型的推理成本攻击（称为“位翻转推理成本攻击”），其目标是受害者模型本身，而不是其输入。具体来说，我们设计了一种简单而有效的方法（称为“BitHydra”）来有效地翻转模型参数的关键部分。该过程由损失函数指导，该函数旨在<EOS>通过高效的关键位搜索算法抑制令牌的概率，从而明确定义攻击目标并实现有效的优化。我们在int8和float 16设置下对11个LLM（参数范围从1.5B到14B）上评估了我们的方法。实验结果表明，只需4个搜索样本和少至3位翻转，BitHydra就可以强制100%的测试提示达到最大生成长度（例如，2048个令牌），突出了其效率，可扩展性和跨看不见的输入的强大可转移性。



## **9. Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models**

分而治之：击败多模式大型语言模型的混合策略 cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.16555v2) [paper-pdf](http://arxiv.org/pdf/2412.16555v2)

**Authors**: Yanxu Mao, Peipei Liu, Tiehan Cui, Zhaoteng Yan, Congying Liu, Datao You

**Abstract**: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.

摘要: 大型语言模型（LLM）因其强大的推理、理解和生成能力而广泛应用于社会各个领域。然而，与这些模型相关的安全问题正变得日益严重。越狱攻击作为检测LLM漏洞的重要方法，已被研究人员探索，他们试图通过各种攻击方法诱导这些模型生成有害内容。然而，现有的越狱方法面临着许多局限性，例如过多的查询次数、越狱模式的覆盖范围有限、攻击成功率低以及评估方法简单化。为了克服这些限制，本文提出了一种多模式越狱方法：JMLLM。该方法集成了多种策略，以跨文本、视觉和听觉方式执行全面的越狱攻击。此外，我们还为多模式越狱研究提供了一个新的全面数据集：TriJail，其中包括所有三种模式的越狱提示。在TriJail数据集和基准数据集AdvBench上进行的实验在13个流行的LLM上进行，展示了先进的攻击成功率和显着减少的时间成本。



## **10. From Evaluation to Defense: Advancing Safety in Video Large Language Models**

从评估到防御：提高视频大型语言模型的安全性 cs.CV

49 pages, 12 figures, 17 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16643v1) [paper-pdf](http://arxiv.org/pdf/2505.16643v1)

**Authors**: Yiwei Sun, Peiqi Jiang, Chuanbin Liu, Luohao Lin, Zhiying Lu, Hongtao Xie

**Abstract**: While the safety risks of image-based large language models have been extensively studied, their video-based counterparts (Video LLMs) remain critically under-examined. To systematically study this problem, we introduce \textbf{VideoSafetyBench (VSB-77k) - the first large-scale, culturally diverse benchmark for Video LLM safety}, which compromises 77,646 video-query pairs and spans 19 principal risk categories across 10 language communities. \textit{We reveal that integrating video modality degrades safety performance by an average of 42.3\%, exposing systemic risks in multimodal attack exploitation.} To address this vulnerability, we propose \textbf{VideoSafety-R1}, a dual-stage framework achieving unprecedented safety gains through two innovations: (1) Alarm Token-Guided Safety Fine-Tuning (AT-SFT) injects learnable alarm tokens into visual and textual sequences, enabling explicit harm perception across modalities via multitask objectives. (2) Then, Safety-Guided GRPO enhances defensive reasoning through dynamic policy optimization with rule-based rewards derived from dual-modality verification. These components synergize to shift safety alignment from passive harm recognition to active reasoning. The resulting framework achieves a 65.1\% improvement on VSB-Eval-HH, and improves by 59.1\%, 44.3\%, and 15.0\% on the image safety datasets MMBench, VLGuard, and FigStep, respectively. \textit{Our codes are available in the supplementary materials.} \textcolor{red}{Warning: This paper contains examples of harmful language and videos, and reader discretion is recommended.}

摘要: 虽然基于图像的大型语言模型的安全风险已经得到了广泛研究，但其基于视频的对应模型（视频LLM）仍然受到严重不足的审查。为了系统性地研究这个问题，我们引入了\textBF{VideoSafetyBench（TSB-77 k）-第一个大规模、文化多样性的视频LLM安全基准}，它包含77，646个视频查询对，涵盖10个语言社区的19个主要风险类别。\textit{我们发现，集成视频模式会使安全性能平均降低42.3%，暴露了多模式攻击利用中的系统性风险。}为了解决这个漏洞，我们提出了\textBF{VideoSafety-R1}，这是一个双阶段框架，通过两项创新实现前所未有的安全收益：（1）警报令牌引导安全微调（AT-SFT）将可学习的警报令牌注入视觉和文本序列中，通过多任务目标实现跨模式的明确伤害感知。(2)然后，安全引导的GRPO通过动态策略优化和双模式验证中的基于规则的奖励来增强防御推理。这些组件协同作用，将安全调整从被动伤害识别转变为主动推理。最终的框架在VSB-Eval-HH上实现了65.1%的改进，在图像安全数据集MMBench、VLGuard和FigStep上分别提高了59.1%、44.3%和15.0%。\texttit {我们的代码可在补充材料中找到。} \textColor{red}{警告：本文包含有害语言和视频的示例，建议读者自行决定。}



## **11. BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization**

BadVLA：通过解耦优化实现对视觉-语言-动作模型的后门攻击 cs.CR

19 pages, 12 figures, 6 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16640v1) [paper-pdf](http://arxiv.org/pdf/2505.16640v1)

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Hechang Wang, Pan Zhou, Lichao Sun

**Abstract**: Vision-Language-Action (VLA) models have advanced robotic control by enabling end-to-end decision-making directly from multimodal inputs. However, their tightly coupled architectures expose novel security vulnerabilities. Unlike traditional adversarial perturbations, backdoor attacks represent a stealthier, persistent, and practically significant threat-particularly under the emerging Training-as-a-Service paradigm-but remain largely unexplored in the context of VLA models. To address this gap, we propose BadVLA, a backdoor attack method based on Objective-Decoupled Optimization, which for the first time exposes the backdoor vulnerabilities of VLA models. Specifically, it consists of a two-stage process: (1) explicit feature-space separation to isolate trigger representations from benign inputs, and (2) conditional control deviations that activate only in the presence of the trigger, while preserving clean-task performance. Empirical results on multiple VLA benchmarks demonstrate that BadVLA consistently achieves near-100% attack success rates with minimal impact on clean task accuracy. Further analyses confirm its robustness against common input perturbations, task transfers, and model fine-tuning, underscoring critical security vulnerabilities in current VLA deployments. Our work offers the first systematic investigation of backdoor vulnerabilities in VLA models, highlighting an urgent need for secure and trustworthy embodied model design practices. We have released the project page at https://badvla-project.github.io/.

摘要: 视觉-语言-动作（VLA）模型通过直接从多模式输入进行端到端决策，实现了先进的机器人控制。然而，它们的紧密耦合架构暴露了新型安全漏洞。与传统的对抗性扰动不同，后门攻击代表了一种更隐蔽、持久且实际上重大的威胁--特别是在新兴的“服务培训”范式下--但在VLA模型的背景下，它在很大程度上尚未被探索。为了弥补这一差距，我们提出了BadVLA，这是一种基于Inbox-Decoupled优化的后门攻击方法，首次暴露了VLA模型的后门漏洞。具体来说，它由两阶段过程组成：（1）显式特征空间分离，以将触发器表示与良性输入隔离，以及（2）仅在触发器存在时激活的条件控制偏差，同时保持干净任务性能。多个VLA基准的经验结果表明，BadVLA始终实现接近100%的攻击成功率，对干净任务准确性的影响最小。进一步的分析证实了它对常见输入扰动、任务传输和模型微调的稳健性，凸显了当前VLA部署中的关键安全漏洞。我们的工作首次对VLA模型中的后门漏洞进行了系统性调查，凸显了对安全且值得信赖的具体模型设计实践的迫切需求。我们已在https://badvla-project.github.io/上发布了项目页面。



## **12. Finetuning-Activated Backdoors in LLMs**

LLM中的微调激活后门 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16567v1) [paper-pdf](http://arxiv.org/pdf/2505.16567v1)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.

摘要: 微调可开放访问的大型语言模型（LLM）已成为实现特定任务性能改进的标准实践。到目前为止，微调一直被认为是一个受控且安全的过程，其中对良性数据集的训练会导致可预测的行为。在本文中，我们首次证明对手可以创建有毒的LLM，这些LLM最初看起来是良性的，但一旦被下游用户微调，就会表现出恶意行为。为此，我们提出的攻击FAB（微调激活后门）通过元学习技术毒害LLM，以模拟下游微调，明确优化微调模型中恶意行为的出现。与此同时，有毒的LLM会被规范化，以保留一般能力，并且在微调之前不会表现出恶意行为。因此，当用户在自己的数据集上微调看似良性的模型时，他们会在不知不觉中触发其隐藏的后门行为。我们展示了FAB在多个LLM和三种目标行为中的有效性：未经请求的广告、拒绝和越狱。此外，我们表明FAB后门对于用户做出的各种微调选择是稳健的（例如，数据集、步骤数、调度程序）。我们的发现挑战了有关微调安全性的普遍假设，揭示了另一个利用LLM复杂性的关键攻击载体。



## **13. CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning**

CTRAP：嵌入崩溃陷阱以保护大型语言模型免受有害的微调 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16559v1) [paper-pdf](http://arxiv.org/pdf/2505.16559v1)

**Authors**: Biao Yi, Tiansheng Huang, Baolei Zhang, Tong Li, Lihai Nie, Zheli Liu, Li Shen

**Abstract**: Fine-tuning-as-a-service, while commercially successful for Large Language Model (LLM) providers, exposes models to harmful fine-tuning attacks. As a widely explored defense paradigm against such attacks, unlearning attempts to remove malicious knowledge from LLMs, thereby essentially preventing them from being used to perform malicious tasks. However, we highlight a critical flaw: the powerful general adaptability of LLMs allows them to easily bypass selective unlearning by rapidly relearning or repurposing their capabilities for harmful tasks. To address this fundamental limitation, we propose a paradigm shift: instead of selective removal, we advocate for inducing model collapse--effectively forcing the model to "unlearn everything"--specifically in response to updates characteristic of malicious adaptation. This collapse directly neutralizes the very general capabilities that attackers exploit, tackling the core issue unaddressed by selective unlearning. We introduce the Collapse Trap (CTRAP) as a practical mechanism to implement this concept conditionally. Embedded during alignment, CTRAP pre-configures the model's reaction to subsequent fine-tuning dynamics. If updates during fine-tuning constitute a persistent attempt to reverse safety alignment, the pre-configured trap triggers a progressive degradation of the model's core language modeling abilities, ultimately rendering it inert and useless for the attacker. Crucially, this collapse mechanism remains dormant during benign fine-tuning, ensuring the model's utility and general capabilities are preserved for legitimate users. Extensive empirical results demonstrate that CTRAP effectively counters harmful fine-tuning risks across various LLMs and attack settings, while maintaining high performance in benign scenarios. Our code is available at https://anonymous.4open.science/r/CTRAP.

摘要: 微调即服务虽然对于大型语言模型（LLM）提供商来说在商业上取得了成功，但会使模型暴露于有害的微调攻击之下。作为一种广泛探索的针对此类攻击的防御范式，取消学习尝试从LLM中删除恶意知识，从而从本质上防止它们被用来执行恶意任务。然而，我们强调了一个关键缺陷：LLM强大的一般适应性使它们能够通过快速重新学习或重新利用其能力来完成有害任务来轻松绕过选择性取消学习。为了解决这个根本限制，我们提出了一种范式转变：我们主张诱导模型崩溃，而不是选择性删除--有效地迫使模型“忘记一切”--特别是为了响应恶意适应的更新。这种崩溃直接抵消了攻击者利用的非常普遍的能力，解决了选择性取消学习未解决的核心问题。我们引入崩溃陷阱（CTRAP）作为有条件地实现这一概念的实用机制。CTRAP嵌入在对齐过程中，预配置模型对后续微调动态的反应。如果微调期间的更新构成了扭转安全对齐的持续尝试，那么预配置的陷阱就会引发模型核心语言建模能力的逐渐退化，最终使其对攻击者变得惰性和无用。至关重要的是，这种崩溃机制在良性微调期间保持休眠状态，确保为合法用户保留模型的实用性和通用功能。广泛的实证结果表明，CTRAP可以有效地应对各种LLM和攻击设置中的有害微调风险，同时在良性场景中保持高性能。我们的代码可在https://anonymous.4open.science/r/CTRAP上获取。



## **14. Implicit Jailbreak Attacks via Cross-Modal Information Concealment on Vision-Language Models**

通过视觉语言模型的跨模式信息隐藏进行隐性越狱攻击 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16446v1) [paper-pdf](http://arxiv.org/pdf/2505.16446v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities. However, the expanded input space introduces new attack surfaces. Previous jailbreak attacks often inject malicious instructions from text into less aligned modalities, such as vision. As MLLMs increasingly incorporate cross-modal consistency and alignment mechanisms, such explicit attacks become easier to detect and block. In this work, we propose a novel implicit jailbreak framework termed IJA that stealthily embeds malicious instructions into images via least significant bit steganography and couples them with seemingly benign, image-related textual prompts. To further enhance attack effectiveness across diverse MLLMs, we incorporate adversarial suffixes generated by a surrogate model and introduce a template optimization module that iteratively refines both the prompt and embedding based on model feedback. On commercial models like GPT-4o and Gemini-1.5 Pro, our method achieves attack success rates of over 90% using an average of only 3 queries.

摘要: 多模式大型语言模型（MLLM）实现强大的跨模式推理能力。然而，扩展的输入空间引入了新的攻击面。之前的越狱攻击经常将文本中的恶意指令注入到不一致的模式中，例如视觉。随着MLLM越来越多地结合跨模式一致性和对齐机制，此类显式攻击变得更容易检测和阻止。在这项工作中，我们提出了一种名为IJA的新型隐式越狱框架，该框架通过最低有效位隐写术将恶意指令秘密地嵌入到图像中，并将其与看似良性的图像相关文本提示相结合。为了进一步增强不同MLLM之间的攻击有效性，我们结合了代理模型生成的对抗性后缀，并引入了模板优化模块，该模块根据模型反馈迭代地细化提示和嵌入。在GPT-4 o和Gemini-1.5 Pro等商业型号上，我们的方法平均只需3个查询即可实现超过90%的攻击成功率。



## **15. Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems**

针对基于R1的检索增强生成系统的思想链中毒攻击 cs.IR

7 pages,3 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16367v1) [paper-pdf](http://arxiv.org/pdf/2505.16367v1)

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan

**Abstract**: Retrieval-augmented generation (RAG) systems can effectively mitigate the hallucination problem of large language models (LLMs),but they also possess inherent vulnerabilities. Identifying these weaknesses before the large-scale real-world deployment of RAG systems is of great importance, as it lays the foundation for building more secure and robust RAG systems in the future. Existing adversarial attack methods typically exploit knowledge base poisoning to probe the vulnerabilities of RAG systems, which can effectively deceive standard RAG models. However, with the rapid advancement of deep reasoning capabilities in modern LLMs, previous approaches that merely inject incorrect knowledge are inadequate when attacking RAG systems equipped with deep reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this paper extracts reasoning process templates from R1-based RAG systems, uses these templates to wrap erroneous knowledge into adversarial documents, and injects them into the knowledge base to attack RAG systems. The key idea of our approach is that adversarial documents, by simulating the chain-of-thought patterns aligned with the model's training signals, may be misinterpreted by the model as authentic historical reasoning processes, thus increasing their likelihood of being referenced. Experiments conducted on the MS MARCO passage ranking dataset demonstrate the effectiveness of our proposed method.

摘要: 检索增强生成（RAG）系统可以有效地缓解大型语言模型（LLM）的幻觉问题，但它们也具有固有的漏洞。在RAG系统大规模现实部署之前识别这些弱点非常重要，因为它为未来构建更安全、更强大的RAG系统奠定了基础。现有的对抗攻击方法通常利用知识库中毒来探测RAG系统的漏洞，这可以有效地欺骗标准RAG模型。然而，随着现代LLM深度推理能力的迅速进步，以前仅仅注入错误知识的方法在攻击配备深度推理能力的RAG系统时是不够的。受LLM深度思维能力的启发，本文从基于R1的RAG系统中提取推理过程模板，使用这些模板将错误知识包装到对抗文档中，并将其注入知识库中以攻击RAG系统。我们方法的关键思想是，通过模拟与模型训练信号一致的思维链模式，对抗性文档可能会被模型误解为真实的历史推理过程，从而增加它们被引用的可能性。在MS MARCO通过排名数据集上进行的实验证明了我们提出的方法的有效性。



## **16. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

PoisonArena：揭露检索增强一代中的竞争中毒攻击 cs.IR

29 pages

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.12574v3) [paper-pdf](http://arxiv.org/pdf/2505.12574v3)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.

摘要: 检索增强生成（RAG）系统，广泛用于改善大型语言模型（LLM）的事实基础，越来越容易受到中毒攻击，其中对手将操纵的内容注入检索器的语料库。虽然以前的研究主要集中在单个攻击者的设置，但现实世界的场景往往涉及多个相互竞争的攻击者，这些攻击者的目标相互冲突。在这项工作中，我们介绍PoisonArena，第一个基准系统地研究和评估竞争中毒攻击在RAG。我们形式化的多攻击者威胁模型，攻击者争夺控制答案相同的查询使用互斥的错误信息。PoisonArena利用Bradley-Terry模型来量化每种方法在此类对抗环境中的竞争有效性。通过对Natural Questions和MS MARCO数据集的广泛实验，我们证明了许多孤立成功的攻击策略在竞争压力下失败。我们的研究结果强调了攻击成功率（SVR）和F1评分等传统评估指标的局限性，并强调了竞争性评估来评估现实世界攻击稳健性的必要性。PoisonArena提供了一个标准化的框架，可以在更现实的多对手条件下基准和开发未来的攻击和防御策略。项目页面：https://github.com/yxf203/PoisonArena。



## **17. Three Minds, One Legend: Jailbreak Large Reasoning Model with Adaptive Stacked Ciphers**

三个意识，一个传奇：具有自适应堆叠密码的越狱大型推理模型 cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16241v1) [paper-pdf](http://arxiv.org/pdf/2505.16241v1)

**Authors**: Viet-Anh Nguyen, Shiqian Zhao, Gia Dao, Runyi Hu, Yi Xie, Luu Anh Tuan

**Abstract**: Recently, Large Reasoning Models (LRMs) have demonstrated superior logical capabilities compared to traditional Large Language Models (LLMs), gaining significant attention. Despite their impressive performance, the potential for stronger reasoning abilities to introduce more severe security vulnerabilities remains largely underexplored. Existing jailbreak methods often struggle to balance effectiveness with robustness against adaptive safety mechanisms. In this work, we propose SEAL, a novel jailbreak attack that targets LRMs through an adaptive encryption pipeline designed to override their reasoning processes and evade potential adaptive alignment. Specifically, SEAL introduces a stacked encryption approach that combines multiple ciphers to overwhelm the models reasoning capabilities, effectively bypassing built-in safety mechanisms. To further prevent LRMs from developing countermeasures, we incorporate two dynamic strategies - random and adaptive - that adjust the cipher length, order, and combination. Extensive experiments on real-world reasoning models, including DeepSeek-R1, Claude Sonnet, and OpenAI GPT-o4, validate the effectiveness of our approach. Notably, SEAL achieves an attack success rate of 80.8% on GPT o4-mini, outperforming state-of-the-art baselines by a significant margin of 27.2%. Warning: This paper contains examples of inappropriate, offensive, and harmful content.

摘要: 最近，与传统的大型语言模型（LLM）相比，大型推理模型（LRM）表现出了更高的逻辑能力，引起了人们的广泛关注。尽管它们的性能令人印象深刻，但更强的推理能力引入更严重的安全漏洞的潜力在很大程度上仍然没有得到充分的探索。现有的越狱方法常常难以平衡有效性与鲁棒性与自适应安全机制。在这项工作中，我们提出了SEAL，这是一种新型越狱攻击，通过自适应加密管道针对LRM，该管道旨在覆盖它们的推理过程并规避潜在的自适应对齐。具体来说，SEAL引入了一种堆叠加密方法，该方法结合了多个密码来压倒模型的推理能力，有效地绕过了内置的安全机制。为了进一步防止LRM制定对策，我们结合了两种动态策略--随机和自适应--来调整密码长度、顺序和组合。对真实世界推理模型（包括DeepSeek-R1、Claude Sonnet和OpenAI GPT-o 4）的广泛实验验证了我们方法的有效性。值得注意的是，SEAL在GPT o 4-mini上的攻击成功率为80.8%，远远超过最先进的基线27.2%。警告：本文包含不恰当、冒犯性和有害内容的示例。



## **18. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

PandaGuard：针对越狱攻击的LLM安全性系统评估 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.13862v2) [paper-pdf](http://arxiv.org/pdf/2505.13862v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.

摘要: 大型语言模型（LLM）已经取得了卓越的能力，但仍然容易受到被称为越狱的对抗性提示的影响，这可能会绕过安全对齐并引发有害的输出。尽管LLM安全研究的努力越来越多，但现有的评估往往是分散的，集中在孤立的攻击或防御技术上，缺乏系统的，可重复的分析。在这项工作中，我们引入了PandaGuard，一个统一的模块化框架，将LLM越狱安全建模为一个由攻击者，防御者和法官组成的多代理系统。我们的框架实现了19种攻击方法和12种防御机制，以及多种判断策略，所有这些都在一个灵活的插件架构中，支持多种LLM接口，多种交互模式和配置驱动的实验，从而增强了可重复性和实际部署。基于这个框架，我们开发了PandaBench，这是一个全面的基准，可评估49个LLM和各种判断方法之间的相互作用，需要超过30亿个代币来执行。我们的广泛评估揭示了对模型漏洞、国防成本-性能权衡和判断一致性的关键见解。我们发现，没有一种防御在所有维度上都是最佳的，而且判断分歧会在安全评估中引入非平凡的方差。我们发布代码、配置和评估结果，以支持LLM安全性方面的透明和可重复研究。



## **19. Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering**

保持安全！针对问题解答中的间接攻击，对大型语言模型上下文中的安全策略保留进行基准测试 cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15805v1) [paper-pdf](http://arxiv.org/pdf/2505.15805v1)

**Authors**: Hwan Chang, Yumin Kim, Yonghyun Jun, Hwanhee Lee

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security.

摘要: 随着大型语言模型（LLM）越来越多地部署在企业和政府等敏感领域，确保它们在上下文中遵守用户定义的安全策略至关重要，尤其是在信息不披露方面。虽然之前的LLM研究重点关注一般安全和社会敏感数据，但仍然缺乏针对攻击的上下文安全保护的大规模基准。为了解决这个问题，我们引入了一个新颖的大规模基准数据集CoPriva，以评估LLM在问答中对上下文保密政策的遵守情况。我们的数据集源自现实背景，包括明确的政策和查询，旨在作为寻求违禁信息的直接和具有挑战性的间接攻击。我们在我们的基准上评估了10个LLM，并揭示了一个重大漏洞：许多模型违反了用户定义的策略并泄露了敏感信息。对于间接攻击，这种故障尤其严重，凸显了当前针对敏感应用的LLM安全调整中的关键差距。我们的分析表明，虽然模型通常可以识别查询的正确答案，但它们很难在生成过程中纳入政策约束。相比之下，它们在明确提示时表现出修改输出的部分能力。我们的研究结果强调迫切需要更强大的方法来保证上下文安全。



## **20. Reverse Engineering Human Preferences with Reinforcement Learning**

利用强化学习反向工程人类偏好 cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15795v1) [paper-pdf](http://arxiv.org/pdf/2505.15795v1)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.

摘要: 大型语言模型（LLM）的能力通常由其他经过训练以预测人类偏好的LLM进行评估。这个框架-被称为LLM作为法官-具有高度可扩展性和相对较低的成本。然而，它也容易受到恶意利用，因为LLM响应可以被调整以过度适应法官的偏好。以前的工作表明，候选人LLM生成的答案可以事后编辑，以最大限度地提高法官LLM分配给他们的分数。在这项研究中，我们采用了一种不同的方法，并使用judge-LLM提供的信号作为奖励，以对抗性地调整模型，这些模型生成旨在提高下游性能的文本前置码。我们发现，使用这些模型流水线化的冻结LLM比现有框架获得更高的LLM评估分数。至关重要的是，与直接干预模型响应的其他框架不同，我们的方法几乎无法检测。我们还证明，当候选LLM和判断LLM被训练期间未使用的模型替换时，调整后的前同步码生成器的有效性会转移。这些发现提出了更可靠的法学硕士作为一个法官的评价设置的设计的重要问题。他们还证明，人类的偏好可以有效地进行逆向工程，通过流水线LLM来优化上游的优化，这种方法可以在对抗性攻击之外的各种任务和领域中找到未来的应用。



## **21. Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval**

通过安全上下文检索针对野外越狱攻击的可扩展防御 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15753v1) [paper-pdf](http://arxiv.org/pdf/2505.15753v1)

**Authors**: Taiye Chen, Zeming Wei, Ang Li, Yisen Wang

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to jailbreaking attacks, wherein adversaries exploit carefully engineered prompts to induce harmful or unethical responses. Such threats have raised critical concerns about the safety and reliability of LLMs in real-world deployment. While existing defense mechanisms partially mitigate such risks, subsequent advancements in adversarial techniques have enabled novel jailbreaking methods to circumvent these protections, exposing the limitations of static defense frameworks. In this work, we explore defending against evolving jailbreaking threats through the lens of context retrieval. First, we conduct a preliminary study demonstrating that even a minimal set of safety-aligned examples against a particular jailbreak can significantly enhance robustness against this attack pattern. Building on this insight, we further leverage the retrieval-augmented generation (RAG) techniques and propose Safety Context Retrieval (SCR), a scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our comprehensive experiments demonstrate how SCR achieves superior defensive performance against both established and emerging jailbreaking tactics, contributing a new paradigm to LLM safety. Our code will be available upon publication.

摘要: 众所周知，大型语言模型（LLM）很容易受到越狱攻击，其中对手利用精心设计的提示来引发有害或不道德的反应。此类威胁引发了人们对LLM在现实世界部署中的安全性和可靠性的严重担忧。虽然现有的防御机制部分减轻了此类风险，但对抗技术的后续进步使新型越狱方法能够规避这些保护，暴露了静态防御框架的局限性。在这项工作中，我们探索通过上下文检索的视角抵御不断变化的越狱威胁。首先，我们进行了一项初步研究，证明即使是针对特定越狱的最少一组安全一致的示例也可以显着增强针对这种攻击模式的鲁棒性。在这一见解的基础上，我们进一步利用检索增强生成（RAG）技术并提出安全上下文检索（SR），这是一种针对LLM越狱的可扩展且强大的保护范式。我们全面的实验展示了可控硅如何在针对既定和新兴越狱策略的情况下实现卓越的防御性能，为LLM安全性贡献了新的范式。我们的代码将在发布后提供。



## **22. Shaping the Safety Boundaries: Understanding and Defending Against Jailbreaks in Large Language Models**

塑造安全边界：理解和防御大型语言模型中的越狱 cs.CL

17 pages, 9 figures

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2412.17034v2) [paper-pdf](http://arxiv.org/pdf/2412.17034v2)

**Authors**: Lang Gao, Jiahui Geng, Xiangliang Zhang, Preslav Nakov, Xiuying Chen

**Abstract**: Jailbreaking in Large Language Models (LLMs) is a major security concern as it can deceive LLMs to generate harmful text. Yet, there is still insufficient understanding of how jailbreaking works, which makes it hard to develop effective defense strategies. We aim to shed more light into this issue: we conduct a detailed large-scale analysis of seven different jailbreak methods and find that these disagreements stem from insufficient observation samples. In particular, we introduce \textit{safety boundary}, and we find that jailbreaks shift harmful activations outside that safety boundary, where LLMs are less sensitive to harmful information. We also find that the low and the middle layers are critical in such shifts, while deeper layers have less impact. Leveraging on these insights, we propose a novel defense called \textbf{Activation Boundary Defense} (ABD), which adaptively constrains the activations within the safety boundary. We further use Bayesian optimization to selectively apply the defense method to the low and the middle layers. Our experiments on several benchmarks show that ABD achieves an average DSR of over 98\% against various forms of jailbreak attacks, with less than 2\% impact on the model's general capabilities.

摘要: 大型语言模型（LLM）中的越狱是一个主要的安全问题，因为它可能会欺骗LLM生成有害文本。然而，人们对越狱的运作方式仍然缺乏足够的了解，这使得制定有效的防御策略变得困难。我们的目标是更多地了解这个问题：我们对七种不同的越狱方法进行了详细的大规模分析，发现这些分歧源于观察样本不足。特别是，我们引入了\textit{safety boundary}，我们发现越狱将有害激活转移到安全边界之外，而LLM对有害信息不太敏感。我们还发现，低层和中层在此类转变中至关重要，而较深层的影响较小。利用这些见解，我们提出了一种名为\textBF{Activation Boundary Defense}（ABD）的新型防御，它自适应地将激活限制在安全边界内。我们进一步使用Bayesian优化来选择性地将防御方法应用于低层和中层。我们在多个基准测试上的实验表明，ABD针对各种形式的越狱攻击，平均DSR超过98%，对模型的一般能力影响不到2%。



## **23. Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses**

压力下的一致：评估LLM防御时知情对手的理由 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15738v1) [paper-pdf](http://arxiv.org/pdf/2505.15738v1)

**Authors**: Xiaoxue Yang, Bozhidar Stevanoski, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs.

摘要: 大型语言模型（LLM）被快速部署在从聊天机器人到代理系统的实际应用中。对齐是用于防御诸如即时注入和越狱等攻击的主要方法之一。最近的防御报告甚至对贪婪坐标梯度（GCG）的攻击成功率（ASR）接近于零，GCG是一种白盒攻击，生成对抗性后缀以诱导攻击者期望的输出。然而，这种在离散令牌上的搜索空间非常大，使得找到成功攻击的任务变得困难。例如，GCG已被证明收敛到局部极小值，使其对初始化选择敏感。在本文中，我们使用一个更明智的威胁模型来评估这些防御系统的面向未来的鲁棒性：可以访问有关对齐过程的一些信息的攻击者。具体来说，我们提出了一种知情白盒攻击，利用中间模型检查点来初始化GCG，每个检查点都充当下一个检查点的垫脚石。我们证明这种方法在最先进的（SOTA）防御和模型中非常有效。我们进一步展示了我们的知情初始化，以优于其他初始化方法，并展示了一种基于梯度的检查点选择策略，以极大地提高攻击性能和效率。重要的是，我们还展示了成功找到通用对抗后缀的方法--在不同输入中有效的单个后缀。我们的结果表明，与之前的观点相反，针对基于SOTA匹配的防御，确实存在有效的对抗性后缀，当对手利用对齐知识时，这些后缀可以通过现有的攻击方法找到，甚至存在通用后缀。总而言之，我们的结果凸显了当前基于环境的方法的脆弱性，以及在测试LLM的安全性时需要考虑更强的威胁模型。



## **24. SQL Injection Jailbreak: A Structural Disaster of Large Language Models**

SQL注入越狱：大型语言模型的结构灾难 cs.CR

Accepted by findings of ACL 2025

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2411.01565v6) [paper-pdf](http://arxiv.org/pdf/2411.01565v6)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: Large Language Models (LLMs) are susceptible to jailbreak attacks that can induce them to generate harmful content. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak methods and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. For open-source models, SIJ achieves near 100% attack success rates on five well-known LLMs on the AdvBench and HEx-PHI, while incurring lower time costs compared to previous methods. For closed-source models, SIJ achieves an average attack success rate over 85% across five models in the GPT and Doubao series. Additionally, SIJ exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple adaptive defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.

摘要: 大型语言模型（LLM）容易受到越狱攻击，从而导致它们生成有害内容。之前的越狱方法主要利用LLM的内部属性或功能，例如基于优化的越狱方法和利用模型上下文学习能力的方法。本文中，我们介绍了一种新颖的越狱方法--SQL注入越狱（SIJ），它针对的是LLM的外部属性，具体来说是LLM构建输入提示的方式。通过将越狱信息注入用户提示中，SIJ成功诱导模型输出有害内容。对于开源模型，SIJ在AdvBench和HEx-PHI上的五个知名LLM上实现了接近100%的攻击成功率，同时与之前的方法相比，时间成本更低。对于闭源型号，SIJ在GPT和抖音系列的五种型号中的平均攻击成功率超过85%。此外，SIJ暴露了LLM中的一个新漏洞，迫切需要缓解。为了解决这个问题，我们提出了一种名为Self-Reminder-Key的简单自适应防御方法来对抗SIJ，并通过实验结果证明其有效性。我们的代码可在https://github.com/weiyezhimeng/SQL-Injection-Jailbreak上获取。



## **25. Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!**

在开源LLM上进行微调时要小心：您的微调数据可能会被秘密窃取！ cs.CL

19 pages

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15656v1) [paper-pdf](http://arxiv.org/pdf/2505.15656v1)

**Authors**: Zhexin Zhang, Yuhao Sun, Junxiao Yang, Shiyao Cui, Hongning Wang, Minlie Huang

**Abstract**: Fine-tuning on open-source Large Language Models (LLMs) with proprietary data is now a standard practice for downstream developers to obtain task-specific LLMs. Surprisingly, we reveal a new and concerning risk along with the practice: the creator of the open-source LLMs can later extract the private downstream fine-tuning data through simple backdoor training, only requiring black-box access to the fine-tuned downstream model. Our comprehensive experiments, across 4 popularly used open-source models with 3B to 32B parameters and 2 downstream datasets, suggest that the extraction performance can be strikingly high: in practical settings, as much as 76.3% downstream fine-tuning data (queries) out of a total 5,000 samples can be perfectly extracted, and the success rate can increase to 94.9% in more ideal settings. We also explore a detection-based defense strategy but find it can be bypassed with improved attack. Overall, we highlight the emergency of this newly identified data breaching risk in fine-tuning, and we hope that more follow-up research could push the progress of addressing this concerning risk. The code and data used in our experiments are released at https://github.com/thu-coai/Backdoor-Data-Extraction.

摘要: 对具有专有数据的开源大型语言模型（LLM）进行微调现在已成为下游开发人员获取特定任务LLM的标准实践。令人惊讶的是，我们在实践中揭示了一个新的且令人担忧的风险：开源LLM的创建者稍后可以通过简单的后门训练提取私有下游微调数据，只需要黑匣子访问微调下游模型。我们对4个常用的3B至32 B参数开源模型和2个下游数据集进行了全面的实验，表明提取性能可以非常高：在实际环境中，总共5，000个样本中，多达76.3%的下游微调数据（查询）可以被完美提取，在更理想的环境中，成功率可以提高到94.9%。我们还探索了基于检测的防御策略，但发现可以通过改进的攻击来绕过它。总体而言，我们强调了这种新发现的数据泄露风险在微调中的紧迫性，我们希望更多的后续研究能够推动解决这一相关风险的进展。我们实验中使用的代码和数据发布在https://github.com/thu-coai/Backdoor-Data-Extraction上。



## **26. SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings**

SEA：通过合成嵌入实现多模式大型语言模型的低资源安全性对齐 cs.CL

Accepted in ACL 2025 Main Track

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2502.12562v2) [paper-pdf](http://arxiv.org/pdf/2502.12562v2)

**Authors**: Weikai Lu, Hao Peng, Huiping Zhuang, Cen Chen, Ziqian Zeng

**Abstract**: Multimodal Large Language Models (MLLMs) have serious security vulnerabilities.While safety alignment using multimodal datasets consisting of text and data of additional modalities can effectively enhance MLLM's security, it is costly to construct these datasets. Existing low-resource security alignment methods, including textual alignment, have been found to struggle with the security risks posed by additional modalities. To address this, we propose Synthetic Embedding augmented safety Alignment (SEA), which optimizes embeddings of additional modality through gradient updates to expand textual datasets. This enables multimodal safety alignment training even when only textual data is available. Extensive experiments on image, video, and audio-based MLLMs demonstrate that SEA can synthesize a high-quality embedding on a single RTX3090 GPU within 24 seconds. SEA significantly improves the security of MLLMs when faced with threats from additional modalities. To assess the security risks introduced by video and audio, we also introduced a new benchmark called VA-SafetyBench. High attack success rates across multiple MLLMs validate its challenge. Our code and data will be available at https://github.com/ZeroNLP/SEA.

摘要: 多模式大型语言模型（MLLM）存在严重的安全漏洞。虽然使用由文本和其他模式数据组成的多模式数据集进行安全对齐可以有效增强MLLM的安全性，但构建这些数据集的成本很高。现有的低资源安全对齐方法（包括文本对齐）被发现难以应对额外模式带来的安全风险。为了解决这个问题，我们提出了合成嵌入增强安全对齐（SEA），它通过梯度更新来优化额外模式的嵌入以扩展文本数据集。即使只有文本数据可用，这也可以实现多模式安全对齐训练。基于图像、视频和音频的MLLM的广泛实验表明，SEA可以在24秒内在单个RTX 3090图形处理器上合成高质量嵌入。SEA在面临来自其他模式的威胁时显着提高了MLLM的安全性。为了评估视频和音频带来的安全风险，我们还引入了名为VA-SafetyBench的新基准。多个MLLM的高攻击成功率证实了其挑战。我们的代码和数据可在https://github.com/ZeroNLP/SEA上获取。



## **27. Silent Leaks: Implicit Knowledge Extraction Attack on RAG Systems through Benign Queries**

Silent Leaks：通过Benign Buttons对RAG系统进行隐式知识提取攻击 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15420v1) [paper-pdf](http://arxiv.org/pdf/2505.15420v1)

**Authors**: Yuhao Wang, Wenjie Qu, Yanze Jiang, Zichen Liu, Yue Liu, Shengfang Zhai, Yinpeng Dong, Jiaheng Zhang

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by incorporating external knowledge bases, but they are vulnerable to privacy risks from data extraction attacks. Existing extraction methods typically rely on malicious inputs such as prompt injection or jailbreaking, making them easily detectable via input- or output-level detection. In this paper, we introduce Implicit Knowledge Extraction Attack (IKEA), which conducts knowledge extraction on RAG systems through benign queries. IKEA first leverages anchor concepts to generate queries with the natural appearance, and then designs two mechanisms to lead to anchor concept thoroughly 'explore' the RAG's privacy knowledge: (1) Experience Reflection Sampling, which samples anchor concepts based on past query-response patterns to ensure the queries' relevance to RAG documents; (2) Trust Region Directed Mutation, which iteratively mutates anchor concepts under similarity constraints to further exploit the embedding space. Extensive experiments demonstrate IKEA's effectiveness under various defenses, surpassing baselines by over 80% in extraction efficiency and 90% in attack success rate. Moreover, the substitute RAG system built from IKEA's extractions consistently outperforms those based on baseline methods across multiple evaluation tasks, underscoring the significant privacy risk in RAG systems.

摘要: 检索增强生成（RAG）系统通过整合外部知识库来增强大型语言模型（LLM），但它们很容易受到数据提取攻击的隐私风险。现有的提取方法通常依赖于恶意输入，例如提示注入或越狱，使得它们可以通过输入或输出级检测轻松检测到。本文引入了隐式知识提取攻击（IKEA），它通过良性查询对RAG系统进行知识提取。宜家首先利用锚概念生成具有自然外观的查询，然后设计了两种机制来引导锚概念彻底“探索”RAG的隐私知识：（1）体验反射采样，基于过去的查询-响应模式对锚概念进行采样，以确保查询与RAG文档的相关性;（2）信任区域定向突变，在相似性约束下迭代突变锚概念，以进一步利用嵌入空间。大量实验证明了宜家在各种防御下的有效性，提取效率超过基线80%，攻击成功率超过基线90%。此外，根据宜家提取物构建的替代RAG系统在多个评估任务中始终优于基于基线方法的系统，这凸显了RAG系统中存在的巨大隐私风险。



## **28. Audio Jailbreak: An Open Comprehensive Benchmark for Jailbreaking Large Audio-Language Models**

Audio Jailbreak：一个用于越狱大型音频语言模型的开放综合基准测试 cs.SD

We release AJailBench, including both static and optimized  adversarial data, to facilitate future research:  https://github.com/mbzuai-nlp/AudioJailbreak

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15406v1) [paper-pdf](http://arxiv.org/pdf/2505.15406v1)

**Authors**: Zirui Song, Qian Jiang, Mingxuan Cui, Mingzhe Li, Lang Gao, Zeyu Zhang, Zixiang Xu, Yanbo Wang, Chenxi Wang, Guangxian Ouyang, Zhenhao Chen, Xiuying Chen

**Abstract**: The rise of Large Audio Language Models (LAMs) brings both potential and risks, as their audio outputs may contain harmful or unethical content. However, current research lacks a systematic, quantitative evaluation of LAM safety especially against jailbreak attacks, which are challenging due to the temporal and semantic nature of speech. To bridge this gap, we introduce AJailBench, the first benchmark specifically designed to evaluate jailbreak vulnerabilities in LAMs. We begin by constructing AJailBench-Base, a dataset of 1,495 adversarial audio prompts spanning 10 policy-violating categories, converted from textual jailbreak attacks using realistic text to speech synthesis. Using this dataset, we evaluate several state-of-the-art LAMs and reveal that none exhibit consistent robustness across attacks. To further strengthen jailbreak testing and simulate more realistic attack conditions, we propose a method to generate dynamic adversarial variants. Our Audio Perturbation Toolkit (APT) applies targeted distortions across time, frequency, and amplitude domains. To preserve the original jailbreak intent, we enforce a semantic consistency constraint and employ Bayesian optimization to efficiently search for perturbations that are both subtle and highly effective. This results in AJailBench-APT, an extended dataset of optimized adversarial audio samples. Our findings demonstrate that even small, semantically preserved perturbations can significantly reduce the safety performance of leading LAMs, underscoring the need for more robust and semantically aware defense mechanisms.

摘要: 大型音频语言模型（LAMs）的兴起带来了潜在的风险，因为它们的音频输出可能包含有害或不道德的内容。然而，目前的研究缺乏一个系统的，定量的评估LAM的安全性，特别是对越狱攻击，这是具有挑战性的，由于语音的时间和语义的性质。为了弥补这一差距，我们引入AJailBench，这是第一个专门用于评估LAM中越狱漏洞的基准测试。我们首先构建AJailBench-Base，这是一个包含1，495个对抗性音频提示的数据集，涵盖10个违反策略的类别，从使用真实文本的文本越狱攻击转换为语音合成。使用该数据集，我们评估了几种最先进的LAM，并发现没有一种在攻击中表现出一致的鲁棒性。为了进一步加强越狱测试并模拟更真实的攻击条件，我们提出了一种生成动态对抗变体的方法。我们的音频微扰工具包（APT）在时间、频率和幅度域中应用有针对性的失真。为了保留最初的越狱意图，我们强制执行语义一致性约束并采用Bayesian优化来有效地搜索微妙且高效的扰动。这会产生AJailBench-APT，这是一个优化的对抗性音频样本的扩展数据集。我们的研究结果表明，即使是很小的、在语义上保留的扰动也会显着降低领先LAM的安全性能，这凸显了对更强大和语义感知的防御机制的需求。



## **29. RePPL: Recalibrating Perplexity by Uncertainty in Semantic Propagation and Language Generation for Explainable QA Hallucination Detection**

RePPL：通过语义传播和语言生成中的不确定性重新校准困惑，以实现可解释QA幻觉检测 cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15386v1) [paper-pdf](http://arxiv.org/pdf/2505.15386v1)

**Authors**: Yiming Huang, Junyan Zhang, Zihao Wang, Biquan Bie, Xuming Hu, Yi R., Fung, Xinlei He

**Abstract**: Large Language Models (LLMs) have become powerful, but hallucinations remain a vital obstacle to their trustworthy use. While previous works improved the capability of hallucination detection by measuring uncertainty, they all lack the ability to explain the provenance behind why hallucinations occur, i.e., which part of the inputs tends to trigger hallucinations. Recent works on the prompt attack indicate that uncertainty exists in semantic propagation, where attention mechanisms gradually fuse local token information into high-level semantics across layers. Meanwhile, uncertainty also emerges in language generation, due to its probability-based selection of high-level semantics for sampled generations. Based on that, we propose RePPL to recalibrate uncertainty measurement by these two aspects, which dispatches explainable uncertainty scores to each token and aggregates in Perplexity-style Log-Average form as total score. Experiments show that our method achieves the best comprehensive detection performance across various QA datasets on advanced models (average AUC of 0.833), and our method is capable of producing token-level uncertainty scores as explanations for the hallucination. Leveraging these scores, we preliminarily find the chaotic pattern of hallucination and showcase its promising usage.

摘要: 大型语言模型（LLM）已经变得强大，但幻觉仍然是其值得信赖使用的重要障碍。虽然之前的作品通过测量不确定性来提高了幻觉检测的能力，但它们都缺乏解释幻觉发生背后来源的能力，即这部分输入往往会引发幻觉。最近关于提示攻击的研究表明，语义传播中存在不确定性，其中注意力机制逐渐将本地令牌信息融合到跨层的高级语义中。与此同时，由于语言生成对采样世代的高级语义基于概率选择，因此也出现了不确定性。在此基础上，我们提出RePPL通过这两个方面重新校准不确定性测量，将可解释的不确定性分数分配到每个代币，并以困惑式的Log-Average形式汇总为总分。实验表明，我们的方法在高级模型上的各种QA数据集中实现了最佳的综合检测性能（平均曲线下面积为0.833），并且我们的方法能够产生符号级的不确定性分数作为幻觉的解释。利用这些分数，我们初步发现了幻觉的混乱模式，并展示了其有希望的用途。



## **30. Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors**

您的语言模型可以像人类一样秘密写作：对LLM生成的文本检测器的对比重述攻击 cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15337v1) [paper-pdf](http://arxiv.org/pdf/2505.15337v1)

**Authors**: Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.

摘要: 学术抄袭等大型语言模型（LLM）的滥用推动了识别LLM生成文本的检测器的发展。为了绕过这些检测器，出现了故意重写这些文本以逃避检测的重述攻击。尽管取得了成功，但现有方法需要大量的数据和计算预算来训练专门的解释器，并且当面对先进的检测算法时，它们的攻击功效会大大降低。为了解决这个问题，我们提出了\textBF{Co} ntrasive\textBF{P}araphrase \textBF{A}ttack（CoPA），这是一种免训练方法，可以使用现成的LLM有效地欺骗文本检测器。第一步是仔细编写指令，鼓励LLM生成更多类似人类的文本。尽管如此，我们观察到LLM固有的统计偏差仍然会导致一些生成的文本携带某些可以被检测器捕获的类似机器的属性。为了克服这个问题，CoPA构建了一个辅助的类似机器的单词分布，与LLM生成的类似人类的分布形成对比。通过在解码过程中从类人分布中减去类机器模式，CoPA能够生成文本检测器难以识别的句子。我们的理论分析表明了拟议攻击的优越性。大量实验验证了CoPA在各种场景中欺骗文本检测器的有效性。



## **31. Towards Zero-Shot Differential Morphing Attack Detection with Multimodal Large Language Models**

利用多模式大型语言模型实现零镜头差异变形攻击检测 cs.CV

Accepted at IEEE International Conference on Automatic Face and  Gesture Recognition (FG 2025)

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15332v1) [paper-pdf](http://arxiv.org/pdf/2505.15332v1)

**Authors**: Ria Shekhawat, Hailin Li, Raghavendra Ramachandra, Sushma Venkatesh

**Abstract**: Leveraging the power of multimodal large language models (LLMs) offers a promising approach to enhancing the accuracy and interpretability of morphing attack detection (MAD), especially in real-world biometric applications. This work introduces the use of LLMs for differential morphing attack detection (D-MAD). To the best of our knowledge, this is the first study to employ multimodal LLMs to D-MAD using real biometric data. To effectively utilize these models, we design Chain-of-Thought (CoT)-based prompts to reduce failure-to-answer rates and enhance the reasoning behind decisions. Our contributions include: (1) the first application of multimodal LLMs for D-MAD using real data subjects, (2) CoT-based prompt engineering to improve response reliability and explainability, (3) comprehensive qualitative and quantitative benchmarking of LLM performance using data from 54 individuals captured in passport enrollment scenarios, and (4) comparative analysis of two multimodal LLMs: ChatGPT-4o and Gemini providing insights into their morphing attack detection accuracy and decision transparency. Experimental results show that ChatGPT-4o outperforms Gemini in detection accuracy, especially against GAN-based morphs, though both models struggle under challenging conditions. While Gemini offers more consistent explanations, ChatGPT-4o is more resilient but prone to a higher failure-to-answer rate.

摘要: 利用多模式大型语言模型（LLM）的力量提供了一种有希望的方法来增强变形攻击检测（MAD）的准确性和可解释性，特别是在现实世界的生物识别应用中。这项工作介绍了使用LLM进行差异变形攻击检测（D-MAD）。据我们所知，这是第一项使用真实生物识别数据将多模式LLM用于D-MAD的研究。为了有效地利用这些模型，我们设计了基于思想链（CoT）的提示，以降低未回答率并增强决策背后的推理。我们的贡献包括：（1）使用真实数据对象首次应用多模式LLM进行D-MAD，（2）基于CoT的提示工程以提高响应可靠性和可解释性，（3）使用护照登记场景中捕获的54名个人的数据对LLM性能进行全面的定性和定量基准测试，（4）两种多模式LLM的比较分析：ChatGPT-4 o和Gemini提供了有关其变形攻击检测准确性和决策透明度的见解。实验结果表明，ChatGPT-4 o在检测准确性方面优于Gemini，尤其是针对基于GAN的变形，尽管这两种模型在具有挑战性的条件下都很困难。虽然Gemini提供了更一致的解释，但ChatGPT-4 o更有弹性，但容易出现更高的失败率。



## **32. Improving LLM First-Token Predictions in Multiple-Choice Question Answering via Prefilling Attack**

通过预填充攻击改进多项选择题回答中的LLM第一令牌预测 cs.CL

13 pages, 5 figures, 7 tables

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15323v1) [paper-pdf](http://arxiv.org/pdf/2505.15323v1)

**Authors**: Silvia Cappelletti, Tobia Poppi, Samuele Poppi, Zheng-Xin Yong, Diego Garcia-Olano, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara

**Abstract**: Large Language Models (LLMs) are increasingly evaluated on multiple-choice question answering (MCQA) tasks using *first-token probability* (FTP), which selects the answer option whose initial token has the highest likelihood. While efficient, FTP can be fragile: models may assign high probability to unrelated tokens (*misalignment*) or use a valid token merely as part of a generic preamble rather than as a clear answer choice (*misinterpretation*), undermining the reliability of symbolic evaluation. We propose a simple solution: the *prefilling attack*, a structured natural-language prefix (e.g., "*The correct option is:*") prepended to the model output. Originally explored in AI safety, we repurpose prefilling to steer the model to respond with a clean, valid option, without modifying its parameters. Empirically, the FTP with prefilling strategy substantially improves accuracy, calibration, and output consistency across a broad set of LLMs and MCQA benchmarks. It outperforms standard FTP and often matches the performance of open-ended generation approaches that require full decoding and external classifiers, while being significantly more efficient. Our findings suggest that prefilling is a simple, robust, and low-cost method to enhance the reliability of FTP-based evaluation in multiple-choice settings.

摘要: 大型语言模型（LLM）越来越多地使用 * 第一令牌概率 *（TP）对多项选择题回答（MCQA）任务进行评估，该概率选择初始令牌可能性最高的答案选项。虽然高效，但RTP可能很脆弱：模型可能会将高概率分配给不相关的标记（* 未对准 *），或者仅将有效标记用作通用前序的一部分，而不是作为明确的答案选择（* 误解 *），从而破坏了符号评估的可靠性。我们提出了一个简单的解决方案：* 预填充攻击 *，结构化自然语言前置（例如，“* 正确的选项是：*”）前置于模型输出。我们最初在人工智能安全性方面进行探索，重新利用预填充，以引导模型以干净、有效的选项做出响应，而无需修改其参数。从经验上看，具有预填充策略的RTP大大提高了一系列LLM和MCQA基准的准确性、校准和输出一致性。它的性能优于标准的RTP，并且通常与需要完全解码和外部分类器的开放式生成方法的性能相匹配，同时效率明显更高。我们的研究结果表明，预填充是一种简单、稳健且低成本的方法，可以增强多项选择设置中基于STP的评估的可靠性。



## **33. Securing RAG: A Risk Assessment and Mitigation Framework**

保护RAG：风险评估和缓解框架 cs.CR

8 pages, 3 figures, Sara Ott and Lukas Ammann contributed equally.  This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.08728v2) [paper-pdf](http://arxiv.org/pdf/2505.08728v2)

**Authors**: Lukas Ammann, Sara Ott, Christoph R. Landolt, Marco P. Lehmann

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as the de facto industry standard for user-facing NLP applications, offering the ability to integrate data without re-training or fine-tuning Large Language Models (LLMs). This capability enhances the quality and accuracy of responses but also introduces novel security and privacy challenges, particularly when sensitive data is integrated. With the rapid adoption of RAG, securing data and services has become a critical priority. This paper first reviews the vulnerabilities of RAG pipelines, and outlines the attack surface from data pre-processing and data storage management to integration with LLMs. The identified risks are then paired with corresponding mitigations in a structured overview. In a second step, the paper develops a framework that combines RAG-specific security considerations, with existing general security guidelines, industry standards, and best practices. The proposed framework aims to guide the implementation of robust, compliant, secure, and trustworthy RAG systems.

摘要: 检索增强生成（RAG）已成为面向用户的NLP应用程序事实上的行业标准，提供集成数据的能力，无需重新训练或微调大型语言模型（LLM）。这种能力增强了响应的质量和准确性，但也带来了新的安全和隐私挑战，特别是在集成敏感数据时。随着RAG的迅速采用，保护数据和服务已成为首要任务。本文首先回顾了RAG管道的漏洞，概述了从数据预处理、数据存储管理到与LLM集成的攻击面。然后，在结构化概述中将识别的风险与相应的缓解措施配对。第二步，本文开发了一个框架，该框架将RAG特定的安全考虑因素与现有的通用安全准则、行业标准和最佳实践相结合。拟议的框架旨在指导稳健、合规、安全且值得信赖的RAG系统的实施。



## **34. Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs**

盲点导航：LVLM敏感语义概念的进化发现 cs.CV

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15265v1) [paper-pdf](http://arxiv.org/pdf/2505.15265v1)

**Authors**: Zihao Pan, Yu Tong, Weibin Wu, Jingyi Wang, Lifeng Chen, Zhe Zhao, Jiajia Wei, Yitong Qiao, Zibin Zheng

**Abstract**: Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research.

摘要: 对抗性攻击旨在生成误导深度模型的恶意输入，但除了导致模型失败之外，它们无法提供某些可解释的信息，例如'\textit{输入中的哪些内容使模型更有可能失败？}”“然而，这些信息对于研究人员专门提高模型稳健性至关重要。最近的研究表明，模型可能对视觉输入中的某些语义（例如“湿”、“雾”）特别敏感，这使得它们容易出错。受此启发，本文对大型视觉语言模型（LVLM）进行了首次探索，发现LVLM在面对图像中的特定语义概念时确实容易产生幻觉和各种错误。为了有效地搜索这些敏感概念，我们集成了大型语言模型（LLM）和文本到图像（T2 I）模型，提出了一种新颖的语义进化框架。随机初始化的语义概念经过基于LLM的交叉和变异操作以形成图像描述，然后由T2 I模型将其转换为LVLM的视觉输入。LVLM在每个输入上的特定任务性能被量化为所涉及语义的适应度分数，并作为奖励信号，以进一步指导LLM探索引发LVLM的概念。对七种主流LVLM和两种多模式任务的广泛实验证明了我们方法的有效性。此外，我们还提供了有关LVLM敏感语义的有趣发现，旨在激发进一步的深入研究。



## **35. From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios**

从言语到碰撞：法学硕士指导的评估和安全关键驾驶场景的对抗生成 cs.AI

New version of the paper

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2502.02145v3) [paper-pdf](http://arxiv.org/pdf/2502.02145v3)

**Authors**: Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, Johannes Betz

**Abstract**: Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.

摘要: 确保自动驾驶汽车的安全需要基于虚拟环境的测试，这取决于安全关键场景的稳健评估和生成。到目前为止，研究人员已经使用基于情景的测试框架，这些框架严重依赖手工制作的场景作为安全指标。为了减少人类解释的工作量并克服这些方法的有限可扩展性，我们将大型语言模型（LLM）与结构化场景解析相结合，并提示工程技术自动评估和生成对安全至关重要的驾驶场景。我们引入了用于场景评估的Cartesian和以自我为中心的提示策略，以及一个对抗生成模块，该模块修改风险诱导车辆（自我攻击者）的轨迹以创建关键场景。我们使用2D仿真框架和多个预先训练的LLM来验证我们的方法。结果表明，该评估模块能够有效地检测碰撞场景，并推断出场景安全性.与此同时，新一代模块识别高风险代理并综合现实的安全关键场景。我们的结论是，LLM配备域知情的提示技术可以有效地评估和生成安全关键的驾驶场景，减少依赖手工制作的指标。我们在https://github.com/TUM-AVS/From-Words-to-Collisions上发布我们的开源代码和场景。



## **36. Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models**

视觉语言模型的少镜头对抗低级微调 cs.LG

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15130v1) [paper-pdf](http://arxiv.org/pdf/2505.15130v1)

**Authors**: Sajjad Ghiasvand, Haniyeh Ehsani Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani

**Abstract**: Vision-Language Models (VLMs) such as CLIP have shown remarkable performance in cross-modal tasks through large-scale contrastive pre-training. To adapt these large transformer-based models efficiently for downstream tasks, Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA have emerged as scalable alternatives to full fine-tuning, especially in few-shot scenarios. However, like traditional deep neural networks, VLMs are highly vulnerable to adversarial attacks, where imperceptible perturbations can significantly degrade model performance. Adversarial training remains the most effective strategy for improving model robustness in PEFT. In this work, we propose AdvCLIP-LoRA, the first algorithm designed to enhance the adversarial robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method formulates adversarial fine-tuning as a minimax optimization problem and provides theoretical guarantees for convergence under smoothness and nonconvex-strong-concavity assumptions. Empirical results across eight datasets using ViT-B/16 and ViT-B/32 models show that AdvCLIP-LoRA significantly improves robustness against common adversarial attacks (e.g., FGSM, PGD), without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA as a practical and theoretically grounded approach for robust adaptation of VLMs in resource-constrained settings.

摘要: 通过大规模对比预训练，CLIP等视觉语言模型（VLM）在跨模式任务中表现出了出色的表现。为了有效地调整这些基于变压器的大型模型以适应下游任务，LoRA等参数高效微调（PEFT）技术已成为完全微调的可扩展替代方案，尤其是在少量场景中。然而，与传统的深度神经网络一样，VLM非常容易受到对抗攻击，其中不可感知的扰动可能会显着降低模型性能。对抗训练仍然是提高PEFT模型稳健性的最有效策略。在这项工作中，我们提出了AdvCLIP-LoRA，这是第一个旨在增强在少数镜头设置中使用LoRA微调的CLIP模型的对抗鲁棒性的算法。我们的方法将对抗性微调表述为极小极大优化问题，并为光滑性和非凸强插值假设下的收敛提供理论保证。使用ViT-B/16和ViT-B/32模型的八个数据集的经验结果表明，AdvCLIP-LoRA显着提高了针对常见对抗攻击（例如，FGSM、PVD），而不会牺牲太多干净的准确性。这些发现凸显了AdvCLIP-LoRA是一种实用且具有理论依据的方法，用于在资源有限的环境中稳健地适应VLM。



## **37. AGENTFUZZER: Generic Black-Box Fuzzing for Indirect Prompt Injection against LLM Agents**

AGENTFUZER：通用黑匣子模糊处理，用于立即间接注射LLM试剂 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.05849v2) [paper-pdf](http://arxiv.org/pdf/2505.05849v2)

**Authors**: Zhun Wang, Vincent Siu, Zhe Ye, Tianneng Shi, Yuzhou Nie, Xuandong Zhao, Chenguang Wang, Wenbo Guo, Dawn Song

**Abstract**: The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.

摘要: 大型语言模型（LLM）强大的规划和推理能力促进了基于代理的系统的开发，这些系统能够利用外部工具并与日益复杂的环境进行交互。然而，这些强大的功能也引入了一个严重的安全风险：间接提示注入，这是一种复杂的攻击载体，通过操纵上下文信息而不是直接用户提示来损害这些代理的核心LLM。在这项工作中，我们提出了一个通用的黑匣子模糊框架AgentXploit，旨在自动发现和利用不同LLM代理之间的间接提示注入漏洞。我们的方法首先构建高质量的初始种子库，然后采用基于蒙特卡洛树搜索（MCTS）的种子选择算法来迭代细化输入，从而最大化发现代理弱点的可能性。我们在AgentDojo和VWA-adv这两个公共基准上评估了AgentXploit，它分别对基于o3-mini和GPT-4 o的代理实现了71%和70%的成功率，几乎是基线攻击性能的两倍。此外，AgentXploit在看不见的任务和内部LLM之间具有很强的可移植性，以及对抗防御的有希望的结果。除了基准评估之外，我们还将我们的攻击应用于现实环境中，成功地误导代理导航到任意URL，包括恶意网站。



## **38. Optimizing Adaptive Attacks against Watermarks for Language Models**

优化针对语言模型水印的自适应攻击 cs.CR

To appear at the International Conference on Machine Learning  (ICML'25)

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2410.02440v2) [paper-pdf](http://arxiv.org/pdf/2410.02440v2)

**Authors**: Abdulrahman Diaa, Toluwani Aremu, Nils Lukas

**Abstract**: Large Language Models (LLMs) can be misused to spread unwanted content at scale. Content watermarking deters misuse by hiding messages in content, enabling its detection using a secret watermarking key. Robustness is a core security property, stating that evading detection requires (significant) degradation of the content's quality. Many LLM watermarking methods have been proposed, but robustness is tested only against non-adaptive attackers who lack knowledge of the watermarking method and can find only suboptimal attacks. We formulate watermark robustness as an objective function and use preference-based optimization to tune adaptive attacks against the specific watermarking method. Our evaluation shows that (i) adaptive attacks evade detection against all surveyed watermarks, (ii) training against any watermark succeeds in evading unseen watermarks, and (iii) optimization-based attacks are cost-effective. Our findings underscore the need to test robustness against adaptively tuned attacks. We release our adaptively optimized paraphrasers at https://github.com/nilslukas/ada-wm-evasion.

摘要: 大型语言模型（LLM）可能会被滥用来大规模传播不需要的内容。内容水印通过在内容中隐藏消息来阻止滥用，从而使用秘密水印密钥进行检测。稳健性是核心安全属性，表明逃避检测需要（显着）降低内容质量。已经提出了许多LLM水印方法，但鲁棒性仅针对缺乏水印方法知识并且只能发现次优攻击的非适应性攻击者进行测试。我们将水印鲁棒性制定为目标函数，并使用基于偏好的优化来调整针对特定水印方法的自适应攻击。我们的评估表明，（i）自适应攻击可以逃避对所有调查的水印的检测，（ii）针对任何水印的训练可以成功地逃避不可见的水印，（iii）基于优化的攻击具有成本效益。我们的发现强调了测试针对自适应调整攻击的稳健性的必要性。我们在https://github.com/nilslukas/ada-wm-evasion上发布了自适应优化的解释。



## **39. AudioJailbreak: Jailbreak Attacks against End-to-End Large Audio-Language Models**

AudioJailbreak：针对端到端大型音频语言模型的越狱攻击 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.14103v2) [paper-pdf](http://arxiv.org/pdf/2505.14103v2)

**Authors**: Guangke Chen, Fu Song, Zhe Zhao, Xiaojun Jia, Yang Liu, Yanchen Qiao, Weizhe Zhang

**Abstract**: Jailbreak attacks to Large audio-language models (LALMs) are studied recently, but they achieve suboptimal effectiveness, applicability, and practicability, particularly, assuming that the adversary can fully manipulate user prompts. In this work, we first conduct an extensive experiment showing that advanced text jailbreak attacks cannot be easily ported to end-to-end LALMs via text-to speech (TTS) techniques. We then propose AudioJailbreak, a novel audio jailbreak attack, featuring (1) asynchrony: the jailbreak audio does not need to align with user prompts in the time axis by crafting suffixal jailbreak audios; (2) universality: a single jailbreak perturbation is effective for different prompts by incorporating multiple prompts into perturbation generation; (3) stealthiness: the malicious intent of jailbreak audios will not raise the awareness of victims by proposing various intent concealment strategies; and (4) over-the-air robustness: the jailbreak audios remain effective when being played over the air by incorporating the reverberation distortion effect with room impulse response into the generation of the perturbations. In contrast, all prior audio jailbreak attacks cannot offer asynchrony, universality, stealthiness, or over-the-air robustness. Moreover, AudioJailbreak is also applicable to the adversary who cannot fully manipulate user prompts, thus has a much broader attack scenario. Extensive experiments with thus far the most LALMs demonstrate the high effectiveness of AudioJailbreak. We highlight that our work peeks into the security implications of audio jailbreak attacks against LALMs, and realistically fosters improving their security robustness. The implementation and audio samples are available at our website https://audiojailbreak.github.io/AudioJailbreak.

摘要: 最近研究了对大型音频语言模型（LALM）的越狱攻击，但它们的有效性、适用性和实用性达到了次优，特别是假设对手可以完全操纵用户提示。在这项工作中，我们首先进行了一项广泛的实验，表明高级文本越狱攻击无法通过文本转语音（TTC）技术轻松移植到端到端LALM。然后，我们提出AudioJailbreak，一种新颖的音频越狱攻击，其特点是：（1）狡猾：越狱音频不需要通过制作后缀的越狱音频在时间轴上与用户提示对齐;（2）通用性：通过将多个提示合并到扰动生成中，单个越狱扰动对不同的提示有效;（3）隐蔽性：越狱音频的恶意意图不会通过提出各种意图隐藏策略来提高受害者的意识;以及（4）空中鲁棒性：越狱音频在空中播放时仍然有效，通过将回响失真效应与房间脉冲响应结合起来扰动的产生。相比之下，所有先前的音频越狱攻击都无法提供灵活性、普遍性、隐蔽性或空中鲁棒性。此外，AudioJailbreak还适用于无法完全操纵用户提示的对手，因此具有更广泛的攻击场景。迄今为止，对大多数LALM的广泛实验证明了AudioJailbreak的高有效性。我们强调，我们的工作探讨了针对LALM的音频越狱攻击的安全影响，并切实促进了其安全稳健性的提高。实现和音频示例可在我们的网站https://audiojailbreak.github.io/AudioJailbreak上获取。



## **40. sudoLLM : On Multi-role Alignment of Language Models**

sudoLLM：关于语言模型的多角色对齐 cs.CL

Under review. Code and data to be released later

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14607v1) [paper-pdf](http://arxiv.org/pdf/2505.14607v1)

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Joy Mahapatra, Utpal Garain

**Abstract**: User authorization-based access privileges are a key feature in many safety-critical systems, but have thus far been absent from the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, and resistance to prompt-based jailbreaking attacks. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs.

摘要: 基于用户授权的访问特权是许多安全关键系统的一个关键功能，但迄今为止在大型语言模型（LLM）领域还没有。在这项工作中，我们从此类访问控制系统中汲取灵感，引入了sudoLLM，这是一种新颖的框架，可以产生多角色对齐的LLM，即负责用户访问权限并按照用户访问权限行事的LLM。sudoLLM将微妙的基于用户的偏见注入到查询中，并训练LLM利用此偏见信号，以便在且仅在用户获得授权的情况下生成敏感信息。我们提出的经验结果表明，这种方法显示出对基于预算的越狱攻击的一致性、概括性和抵抗性大幅提高。语言建模目标和安全对齐之间的持续紧张关系（通常被用来越狱LLM）在注入的偏见信号的帮助下在一定程度上得到了解决。我们的框架旨在作为额外的安全层，并补充现有的护栏机制，通过LLM增强端到端安全性。



## **41. MrGuard: A Multilingual Reasoning Guardrail for Universal LLM Safety**

MrGuard：通用LLM安全的多语言推理保障 cs.CL

Preprint

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2504.15241v2) [paper-pdf](http://arxiv.org/pdf/2504.15241v2)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual settings, where multilingual safety-aligned data is often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we introduce a multilingual guardrail with reasoning for prompt classification. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-based Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail, MrGuard, consistently outperforms recent baselines across both in-domain and out-of-domain languages by more than 15%. We also evaluate MrGuard's robustness to multilingual variations, such as code-switching and low-resource language distractors in the prompt, and demonstrate that it preserves safety judgments under these challenging conditions. The multilingual reasoning capability of our guardrail enables it to generate explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.

摘要: 大型语言模型（LLM）容易受到诸如越狱之类的对抗性攻击，这可能会引发有害或不安全的行为。这种漏洞在多语言环境中会加剧，其中多语言安全一致的数据通常是有限的。因此，开发一个能够检测和过滤不同语言的不安全内容的护栏对于在现实世界的应用程序中部署LLM至关重要。在这项工作中，我们介绍了一种多语言护栏，具有快速分类的推理。我们的方法包括：（1）综合多语言数据生成，融合了文化和语言上的细微差别，（2）监督式微调，以及（3）进一步提高性能的基于课程的组相对政策优化（GRPO）框架。实验结果表明，我们的多语言护栏MrGuard在域内和域外语言中的表现始终优于最近的基线15%以上。我们还评估了MrGuard对多语言变体（例如提示中的代码切换和低资源语言干扰因素）的稳健性，并证明它在这些具有挑战性的条件下保留了安全判断。我们护栏的多语言推理能力使其能够生成解释，这对于理解多语言内容审核中的特定语言风险和歧义特别有用。



## **42. Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in Multilingual LLMs**

Char-mander使用mBackdoor！多语言LLM中的跨语言后门攻击研究 cs.CL

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2502.16901v2) [paper-pdf](http://arxiv.org/pdf/2502.16901v2)

**Authors**: Himanshu Beniwal, Sailesh Panda, Birudugadda Srivibhav, Mayank Singh

**Abstract**: We explore \textbf{C}ross-lingual \textbf{B}ackdoor \textbf{AT}tacks (X-BAT) in multilingual Large Language Models (mLLMs), revealing how backdoors inserted in one language can automatically transfer to others through shared embedding spaces. Using toxicity classification as a case study, we demonstrate that attackers can compromise multilingual systems by poisoning data in a single language, with rare and high-occurring tokens serving as specific, effective triggers. Our findings expose a critical vulnerability that influences the model's architecture, resulting in a concealed backdoor effect during the information flow. Our code and data are publicly available https://github.com/himanshubeniwal/X-BAT.

摘要: 我们探索了多语言大型语言模型（mLLM）中的\textBF{C}ross-lingual \textBF{B}ackdoor \textBF{AT}tacks（X-BAT），揭示了插入一种语言的后门如何通过共享嵌入空间自动传输到其他语言。使用毒性分类作为案例研究，我们证明攻击者可以通过毒害单一语言的数据来危害多语言系统，其中罕见且高出现的标记充当特定、有效的触发器。我们的研究结果暴露了影响模型架构的一个关键漏洞，导致信息流期间隐藏的后门效应。我们的代码和数据可在https://github.com/himanshubeniwal/X-BAT上公开获取。



## **43. Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders**

绝命毒师代币：使用稀疏自动编码器去规范化LLM cs.CL

Preprint: 19 pages, 7 figures, 1 table

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14536v1) [paper-pdf](http://arxiv.org/pdf/2505.14536v1)

**Authors**: Agam Goyal, Vedant Rathi, William Yeh, Yian Wang, Yuen Chen, Hari Sundaram

**Abstract**: Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment.

摘要: 大型语言模型（LLM）现在在面向用户的应用程序中无处不在，但它们仍然会产生不受欢迎的有毒输出，包括脏话、粗俗和贬损言论。尽管存在多种解毒方法，但大多数都适用于广泛的、表面的修复，因此很容易被越狱攻击规避。在本文中，我们利用稀疏自动编码器（SAEs）来识别模型剩余流中与毒性相关的方向，并使用相应的解码器载体执行有针对性的激活引导。我们引入了三层转向攻击性，并在GPT-2 Small和Gemma-2-2B上对其进行了评估，揭示了毒性降低和语言流利性之间的权衡。在更强的引导强度下，这些因果干预措施在将毒性降低高达20%方面超过了竞争基线，尽管根据攻击性的不同，GPT-2 Small的流畅性可能会显着下降。至关重要的是，转向后的标准NLP基准分数保持稳定，这表明模型的知识和一般能力得到了保留。我们进一步表明，更广泛的严重不良事件中的特征分裂会阻碍安全干预，强调了解开特征学习的重要性。我们的研究结果强调了LLM解毒基于CAE的因果干预措施的前景和当前的局限性，进一步为更安全的语言模型部署提出了实用指南。



## **44. Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in MLLM-Powered Mobile GUI Agents**

Hidden Ghost Hand：揭露MLLM支持的移动图形用户界面代理中的后门漏洞 cs.CL

25 pages, 10 figures, 12 Tables

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14418v1) [paper-pdf](http://arxiv.org/pdf/2505.14418v1)

**Authors**: Pengzhou Cheng, Haowen Hu, Zheng Wu, Zongru Wu, Tianjie Ju, Daizong Ding, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Graphical user interface (GUI) agents powered by multimodal large language models (MLLMs) have shown greater promise for human-interaction. However, due to the high fine-tuning cost, users often rely on open-source GUI agents or APIs offered by AI providers, which introduces a critical but underexplored supply chain threat: backdoor attacks. In this work, we first unveil that MLLM-powered GUI agents naturally expose multiple interaction-level triggers, such as historical steps, environment states, and task progress. Based on this observation, we introduce AgentGhost, an effective and stealthy framework for red-teaming backdoor attacks. Specifically, we first construct composite triggers by combining goal and interaction levels, allowing GUI agents to unintentionally activate backdoors while ensuring task utility. Then, we formulate backdoor injection as a Min-Max optimization problem that uses supervised contrastive learning to maximize the feature difference across sample classes at the representation space, improving flexibility of the backdoor. Meanwhile, it adopts supervised fine-tuning to minimize the discrepancy between backdoor and clean behavior generation, enhancing effectiveness and utility. Extensive evaluations of various agent models in two established mobile benchmarks show that AgentGhost is effective and generic, with attack accuracy that reaches 99.7\% on three attack objectives, and shows stealthiness with only 1\% utility degradation. Furthermore, we tailor a defense method against AgentGhost that reduces the attack accuracy to 22.1\%. Our code is available at \texttt{anonymous}.

摘要: 由多模式大型语言模型（MLLM）支持的图形用户界面（图形用户界面）代理在人际交互方面表现出了更大的前景。然而，由于微调成本很高，用户通常依赖人工智能提供商提供的开源图形界面代理或API，这引入了一个关键但未充分开发的供应链威胁：后门攻击。在这项工作中，我们首先揭示了基于MLLM的图形用户界面代理自然暴露多个交互级触发器，例如历史步骤、环境状态和任务进度。基于这一观察，我们引入了AgentGhost，这是一个用于红色团队后门攻击的有效且隐蔽的框架。具体来说，我们首先通过结合目标和交互级别来构建复合触发器，允许图形用户界面代理无意中激活后门，同时确保任务实用性。然后，我们将后门注入制定为Min-Max优化问题，该问题使用监督对比学习来最大化表示空间中样本类之间的特征差异，从而提高后门的灵活性。同时，它采用监督式微调，以最大限度地减少后门和干净行为生成之间的差异，提高有效性和实用性。对两个已建立的移动基准测试中各种代理模型的广泛评估表明，AgentGhost有效且通用，在三个攻击目标上的攻击准确率达到99.7%，并且表现出隐蔽性，仅使用1%的效用下降。此外，我们针对AgentGhost定制了一种防御方法，将攻击准确率降低至22.1%。我们的代码可在\textttt {anonymous}上获取。



## **45. Is Your Prompt Safe? Investigating Prompt Injection Attacks Against Open-Source LLMs**

您的提示安全吗？调查针对开源LLM的即时注入攻击 cs.CR

8 pages, 3 figures, EMNLP 2025 under review

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14368v1) [paper-pdf](http://arxiv.org/pdf/2505.14368v1)

**Authors**: Jiawen Wang, Pritha Gupta, Ivan Habernal, Eyke Hüllermeier

**Abstract**: Recent studies demonstrate that Large Language Models (LLMs) are vulnerable to different prompt-based attacks, generating harmful content or sensitive information. Both closed-source and open-source LLMs are underinvestigated for these attacks. This paper studies effective prompt injection attacks against the $\mathbf{14}$ most popular open-source LLMs on five attack benchmarks. Current metrics only consider successful attacks, whereas our proposed Attack Success Probability (ASP) also captures uncertainty in the model's response, reflecting ambiguity in attack feasibility. By comprehensively analyzing the effectiveness of prompt injection attacks, we propose a simple and effective hypnotism attack; results show that this attack causes aligned language models, including Stablelm2, Mistral, Openchat, and Vicuna, to generate objectionable behaviors, achieving around $90$% ASP. They also indicate that our ignore prefix attacks can break all $\mathbf{14}$ open-source LLMs, achieving over $60$% ASP on a multi-categorical dataset. We find that moderately well-known LLMs exhibit higher vulnerability to prompt injection attacks, highlighting the need to raise public awareness and prioritize efficient mitigation strategies.

摘要: 最近的研究表明，大型语言模型（LLM）容易受到不同的基于预算的攻击，从而生成有害内容或敏感信息。闭源和开源LLM的这些攻击都没有得到充分的调查。本文在五个攻击基准上研究了针对$\mathBF{14}$最受欢迎的开源LLM的有效即时注入攻击。当前的指标仅考虑成功的攻击，而我们提出的攻击成功概率（ISP）还捕捉了模型响应中的不确定性，反映了攻击可行性的模糊性。通过全面分析提示注入攻击的有效性，我们提出了一种简单有效的催眠攻击;结果表明，这种攻击会导致包括Stablelm 2、Mistral、Openchat和Vicuna在内的对齐语言模型产生令人反感的行为，实现了约90美元$%的目标。它们还表明，我们的忽略前置攻击可以破坏所有$\mathBF{14}$开源LLM，在多类别数据集上实现超过60美元$%的平均利润。我们发现，中等知名的LLM对引发注入攻击的脆弱性更高，这凸显了提高公众意识并优先考虑有效的缓解策略的必要性。



## **46. Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation**

通过弱到强的知识蒸馏消除LLM的后门攻击 cs.CL

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2410.14425v2) [paper-pdf](http://arxiv.org/pdf/2410.14425v2)

**Authors**: Shuai Zhao, Xiaobao Wu, Cong-Duy Nguyen, Yanhao Jia, Meihuizi Jia, Yichao Feng, Luu Anh Tuan

**Abstract**: Parameter-efficient fine-tuning (PEFT) can bridge the gap between large language models (LLMs) and downstream tasks. However, PEFT has been proven vulnerable to malicious attacks. Research indicates that poisoned LLMs, even after PEFT, retain the capability to activate internalized backdoors when input samples contain predefined triggers. In this paper, we introduce a novel weak-to-strong unlearning algorithm to defend against backdoor attacks based on feature alignment knowledge distillation, named W2SDefense. Specifically, we first train a small-scale language model through full-parameter fine-tuning to serve as the clean teacher model. Then, this teacher model guides the large-scale poisoned student model in unlearning the backdoor, leveraging PEFT. Theoretical analysis suggests that W2SDefense has the potential to enhance the student model's ability to unlearn backdoor features, preventing the activation of the backdoor. We conduct comprehensive experiments on three state-of-the-art large language models and several different backdoor attack algorithms. Our empirical results demonstrate the outstanding performance of W2SDefense in defending against backdoor attacks without compromising model performance.

摘要: 参数高效微调（PEFT）可以弥合大型语言模型（LLM）和下游任务之间的差距。然而，PEFT已被证明容易受到恶意攻击。研究表明，即使在PEFT之后，中毒的LLM也保留在输入样本包含预定义触发器时激活内化后门的能力。本文引入了一种基于特征对齐知识提炼的新型弱到强去学习算法，名为W2 SDefense，以抵御后门攻击。具体来说，我们首先通过全参数微调训练小规模语言模型，作为干净教师模型。然后，这个教师模型引导大规模中毒学生模型利用PEFT摆脱后门。理论分析表明，W2 SDenance有潜力增强学生模型忘记后门功能的能力，防止后门被激活。我们对三种最先进的大型语言模型和几种不同的后门攻击算法进行了全面的实验。我们的实证结果证明了W2 SDenance在防御后门攻击方面具有出色的性能，而不影响模型性能。



## **47. Exploring Jailbreak Attacks on LLMs through Intent Concealment and Diversion**

探索通过意图隐瞒和转移对LLM的越狱攻击 cs.CR

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14316v1) [paper-pdf](http://arxiv.org/pdf/2505.14316v1)

**Authors**: Tiehan Cui, Yanxu Mao, Peipei Liu, Congying Liu, Datao You

**Abstract**: Although large language models (LLMs) have achieved remarkable advancements, their security remains a pressing concern. One major threat is jailbreak attacks, where adversarial prompts bypass model safeguards to generate harmful or objectionable content. Researchers study jailbreak attacks to understand security and robustness of LLMs. However, existing jailbreak attack methods face two main challenges: (1) an excessive number of iterative queries, and (2) poor generalization across models. In addition, recent jailbreak evaluation datasets focus primarily on question-answering scenarios, lacking attention to text generation tasks that require accurate regeneration of toxic content. To tackle these challenges, we propose two contributions: (1) ICE, a novel black-box jailbreak method that employs Intent Concealment and divErsion to effectively circumvent security constraints. ICE achieves high attack success rates (ASR) with a single query, significantly improving efficiency and transferability across different models. (2) BiSceneEval, a comprehensive dataset designed for assessing LLM robustness in question-answering and text-generation tasks. Experimental results demonstrate that ICE outperforms existing jailbreak techniques, revealing critical vulnerabilities in current defense mechanisms. Our findings underscore the necessity of a hybrid security strategy that integrates predefined security mechanisms with real-time semantic decomposition to enhance the security of LLMs.

摘要: 尽管大型语言模型（LLM）取得了显着的进步，但其安全性仍然是一个紧迫的问题。一个主要威胁是越狱攻击，其中敌对性会促使绕过模型保护措施来生成有害或令人反感的内容。研究人员研究越狱攻击以了解LLM的安全性和稳健性。然而，现有的越狱攻击方法面临两个主要挑战：（1）迭代查询数量过多，（2）模型之间的概括性较差。此外，最近的越狱评估数据集主要关注问答场景，缺乏对需要准确再生有毒内容的文本生成任务的关注。为了应对这些挑战，我们提出了两个贡献：（1）ICE，一种新的黑盒越狱方法，采用意图隐藏和分裂，以有效地规避安全约束。ICE通过单个查询实现了高攻击成功率（ASR），显著提高了效率和跨不同模型的可移植性。(2)BiSceneEval是一个综合数据集，旨在评估LLM在问答和文本生成任务中的鲁棒性。实验结果表明，ICE优于现有的越狱技术，揭示了当前防御机制中的关键漏洞。我们的研究结果强调了混合安全策略的必要性，该策略将预定义的安全机制与实时语义分解集成在一起，以增强LLM的安全性。



## **48. Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs**

语音灵活控制的通用声学对抗攻击-LLM cs.CL

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14286v1) [paper-pdf](http://arxiv.org/pdf/2505.14286v1)

**Authors**: Rao Ma, Mengjie Qian, Vyas Raina, Mark Gales, Kate Knill

**Abstract**: The combination of pre-trained speech encoders with large language models has enabled the development of speech LLMs that can handle a wide range of spoken language processing tasks. While these models are powerful and flexible, this very flexibility may make them more vulnerable to adversarial attacks. To examine the extent of this problem, in this work we investigate universal acoustic adversarial attacks on speech LLMs. Here a fixed, universal, adversarial audio segment is prepended to the original input audio. We initially investigate attacks that cause the model to either produce no output or to perform a modified task overriding the original prompt. We then extend the nature of the attack to be selective so that it activates only when specific input attributes, such as a speaker gender or spoken language, are present. Inputs without the targeted attribute should be unaffected, allowing fine-grained control over the model outputs. Our findings reveal critical vulnerabilities in Qwen2-Audio and Granite-Speech and suggest that similar speech LLMs may be susceptible to universal adversarial attacks. This highlights the need for more robust training strategies and improved resistance to adversarial attacks.

摘要: 预训练的语音编码器与大型语言模型的结合使得能够开发出可以处理广泛口语处理任务的语音LLM。虽然这些模型强大且灵活，但这种灵活性可能使它们更容易受到对抗攻击。为了研究这个问题的严重程度，在这项工作中，我们研究了对语音LLM的普遍声学对抗攻击。这里，固定的、通用的、对抗性的音频段被预先添加到原始输入音频上。我们最初调查导致模型不产生输出或执行覆盖原始提示的修改任务的攻击。然后，我们将攻击的性质扩展为选择性，以便只有在特定输入属性（例如说话者性别或口语）存在时，它才会激活。没有目标属性的输入应该不受影响，允许对模型输出进行细粒度控制。我们的研究结果揭示了Qwen 2-Audio和Granite-Speech中的关键漏洞，并表明类似的语音LLM可能容易受到普遍对抗攻击。这凸显了需要更强大的训练策略和提高对对抗性攻击的抵抗力。



## **49. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

针对基于LLM的多代理系统的IP泄露攻击 cs.CR

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.12442v2) [paper-pdf](http://arxiv.org/pdf/2505.12442v2)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.

摘要: 大型语言模型（LLM）的快速发展导致了通过协作执行复杂任务的多智能体系统（MAS）的出现。然而，MAS的复杂性质，包括其架构和代理交互，引发了有关知识产权（IP）保护的严重担忧。本文介绍MASLEAK，这是一种新型攻击框架，旨在从MAS应用程序中提取敏感信息。MASLEAK针对的是实用的黑匣子设置，其中对手不了解MAS架构或代理配置。对手只能通过其公共API与MAS交互，提交攻击查询$q$并观察最终代理的输出。受计算机蠕虫传播和感染脆弱网络主机的方式的启发，MASLEAK精心设计了对抗性查询$q$，以引发、传播和保留每个MAS代理的响应，这些响应揭示了全套专有组件，包括代理数量、系统布局、系统提示、任务指令和工具使用。我们构建了包含810个应用程序的第一个MAS应用程序合成数据集，并根据现实世界的MAS应用程序（包括Coze和CrewAI）评估MASLEAK。MASLEAK在提取MAS IP方面实现了高准确性，系统提示和任务指令的平均攻击成功率为87%，大多数情况下系统架构的平均攻击成功率为92%。最后，我们讨论了我们发现的影响和潜在的防御措施。



## **50. "Haet Bhasha aur Diskrimineshun": Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs**

“Haet Bhasha aur rimineshun”：代码混合印度式英语到红队法学硕士中的语音扰动 cs.CL

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14226v1) [paper-pdf](http://arxiv.org/pdf/2505.14226v1)

**Authors**: Darpan Aswal, Siddharth D Jaiswal

**Abstract**: Large Language Models (LLMs) have become increasingly powerful, with multilingual and multimodal capabilities improving by the day. These models are being evaluated through audits, alignment studies and red-teaming efforts to expose model vulnerabilities towards generating harmful, biased and unfair content. Existing red-teaming efforts have previously focused on the English language, using fixed template-based attacks; thus, models continue to be susceptible to multilingual jailbreaking strategies, especially in the multimodal context. In this study, we introduce a novel strategy that leverages code-mixing and phonetic perturbations to jailbreak LLMs for both text and image generation tasks. We also introduce two new jailbreak strategies that show higher effectiveness than baseline strategies. Our work presents a method to effectively bypass safety filters in LLMs while maintaining interpretability by applying phonetic misspellings to sensitive words in code-mixed prompts. Our novel prompts achieve a 99% Attack Success Rate for text generation and 78% for image generation, with Attack Relevance Rate of 100% for text generation and 95% for image generation when using the phonetically perturbed code-mixed prompts. Our interpretability experiments reveal that phonetic perturbations impact word tokenization, leading to jailbreak success. Our study motivates increasing the focus towards more generalizable safety alignment for multilingual multimodal models, especially in real-world settings wherein prompts can have misspelt words.

摘要: 大型语言模型（LLM）变得越来越强大，多语言和多模式能力日益提高。这些模型正在通过审计、对齐研究和红色团队工作进行评估，以暴露模型在生成有害、偏见和不公平内容方面的弱点。现有的红色团队工作此前主要集中在英语上，使用固定的基于模板的攻击;因此，模型仍然容易受到多语言越狱策略的影响，尤其是在多模式环境中。在这项研究中，我们引入了一种新颖的策略，该策略利用代码混合和语音扰动来越狱文本和图像生成任务的LLM。我们还引入了两种新的越狱策略，它们的有效性比基线策略更高。我们的工作提出了一种方法，可以有效地绕过LLM中的安全过滤器，同时通过对代码混合提示中的敏感词应用语音拼写错误来保持可解释性。当使用语音干扰的代码混合提示时，我们的新颖提示的文本生成攻击成功率为99%，图像生成攻击成功率为78%，文本生成攻击相关率为100%，图像生成攻击相关率为95%。我们的可解释性实验表明，语音扰动会影响单词符号化，从而导致越狱成功。我们的研究促使人们更加关注多语言多模式模型的更通用的安全对齐，特别是在提示可能有拼写错误单词的现实环境中。



