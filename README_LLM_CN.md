# Latest Large Language Model Attack Papers
**update at 2025-03-19 10:36:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

TAROT：使用策略优化方法的面向任务的作者混淆 cs.CL

Accepted to the NAACL PrivateNLP 2025 Workshop

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2407.21630v2) [paper-pdf](http://arxiv.org/pdf/2407.21630v2)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduces the accuracy of attackers while preserving utility. We make our code and models publicly available.

摘要: 作者身份混淆旨在通过改变与文本作者相关的写作风格、词汇、句法和其他语言特征来掩盖作者在文本中的身份。这一改变需要平衡隐私和效用。虽然强大的混淆技术可以有效地隐藏作者的身份，但它们往往会降低文本的质量和对预期目的的有用性。相反，保持高实用性往往会提供不充分的隐私，使对手更容易解除作者的匿名。因此，在这两个相互冲突的目标之间实现最佳权衡至关重要。在本文中，我们提出了一种新的无监督作者身份混淆方法--TAROT：基于策略优化的面向任务的作者身份混淆方法，其目标是通过重新生成考虑下游效用的整个文本来优化隐私和效用之间的权衡。我们的方法利用策略优化作为小语言模型上的微调范式，以便通过保留作者身份和下游任务效用来重写文本。我们表明，我们的方法在很大程度上降低了攻击者的准确性，同时保持了实用性。我们公开我们的代码和模型。



## **2. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

针对越狱攻击的多模式大型语言模型对抗训练 cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.04833v2) [paper-pdf](http://arxiv.org/pdf/2503.04833v2)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.

摘要: 多通道大语言模型在跨通道理解和生成任务方面取得了显著进展。然而，它们仍然容易受到越狱攻击，在越狱攻击中，精心设计的扰动绕过安全护栏，引发有害输出。在这篇文章中，我们提出了在MLLM训练阶段为防御越狱攻击而定制的第一个对抗性训练(AT)范例。将传统的AT扩展到这一领域会带来两个关键挑战：有效地调整大量参数和确保对跨多个通道的攻击的健壮性。为了应对这些挑战，我们引入了投影层对抗对手训练(ProEAT)，这是一个端到端的AT框架。ProEAT结合了基于投影仪的对抗性训练体系结构，通过将对抗性训练集中在轻量级投影器层而不是整个模型上，在保持计算可行性的同时有效地处理大规模参数；此外，我们设计了动态权重调整机制，基于任务需求优化损失函数的权重分配，从而简化了调整过程。为了提高防御性能，我们提出了一种跨视觉和文本模式的联合优化策略，确保对来自任何一种模式的越狱攻击具有强大的抵抗力。在三种主流MLLMS上对五种主要的越狱攻击方法进行了广泛的实验，证明了该方法的有效性。ProEAT实现了最先进的防御性能，在文本和图像模式中的表现比现有基线平均高出34%，而干净的准确性仅降低了1%。此外，对真实世界体现的智能系统的评估突出了我们框架的实用适用性，为开发更安全可靠的多式联运系统铺平了道路。



## **3. Survey of Adversarial Robustness in Multimodal Large Language Models**

多模式大型语言模型中的对抗鲁棒性研究 cs.CV

9 pages

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13962v1) [paper-pdf](http://arxiv.org/pdf/2503.13962v1)

**Authors**: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.

摘要: 多模式大语言模型(MLLM)通过促进跨不同模式的集成理解，包括文本、图像、视频、音频和语音，在人工智能中表现出出色的性能。然而，它们在实际应用程序中的部署引发了人们对可能危及其安全性和可靠性的对抗性漏洞的严重担忧。与单模模型不同，由于各通道之间的相互依赖关系，最大似然模型面临着独特的挑战，这使得它们容易受到特定通道的威胁和跨通道的对抗性操作。本文综述了MLLMS的对抗稳健性，涵盖了不同的模式。我们首先概述MLLMS和针对每种模式量身定做的对抗性攻击分类。接下来，我们回顾了用于评估MLLMS稳健性的关键数据集和评估指标。在此之后，我们提供了针对不同模式的MLLM的攻击的深入回顾。我们的调查还确定了关键挑战，并提出了前景看好的未来研究方向。



## **4. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

AttackEval：如何评估越狱攻击对大型语言模型的有效性 cs.CL

Accepted by ACM SIGKDD Explorations 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2401.09002v6) [paper-pdf](http://arxiv.org/pdf/2401.09002v6)

**Authors**: Dong Shu, Chong Zhang, Mingyu Jin, Zihao Zhou, Lingyao Li, Yongfeng Zhang

**Abstract**: Jailbreak attacks represent one of the most sophisticated threats to the security of large language models (LLMs). To deal with such risks, we introduce an innovative framework that can help evaluate the effectiveness of jailbreak attacks on LLMs. Unlike traditional binary evaluations focusing solely on the robustness of LLMs, our method assesses the attacking prompts' effectiveness. We present two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework uses a scoring range from 0 to 1, offering unique perspectives and allowing for the assessment of attack effectiveness in different scenarios. Additionally, we develop a comprehensive ground truth dataset specifically tailored for jailbreak prompts. This dataset is a crucial benchmark for our current study and provides a foundational resource for future research. By comparing with traditional evaluation methods, our study shows that the current results align with baseline metrics while offering a more nuanced and fine-grained assessment. It also helps identify potentially harmful attack prompts that might appear harmless in traditional evaluations. Overall, our work establishes a solid foundation for assessing a broader range of attack prompts in prompt injection.

摘要: 越狱攻击是大型语言模型(LLM)安全面临的最复杂的威胁之一。为了应对这样的风险，我们引入了一个创新的框架，可以帮助评估越狱攻击对低收入者的有效性。与传统的只关注LLMS健壮性的二值评估不同，我们的方法评估了攻击提示的有效性。我们提出了两种不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供独特的视角，并允许在不同情况下评估攻击效果。此外，我们还开发了专门为越狱提示量身定做的全面地面事实数据集。该数据集是我们当前研究的重要基准，并为未来的研究提供了基础资源。通过与传统评估方法的比较，我们的研究表明，当前的结果与基线度量一致，同时提供了更细微和细粒度的评估。它还有助于识别在传统评估中可能看起来无害的潜在有害攻击提示。总体而言，我们的工作为在快速注入中评估更广泛的攻击提示奠定了坚实的基础。



## **5. Web Artifact Attacks Disrupt Vision Language Models**

网络收件箱攻击扰乱视觉语言模型 cs.CV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13652v1) [paper-pdf](http://arxiv.org/pdf/2503.13652v1)

**Authors**: Maan Qraitem, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Vision-language models (VLMs) (e.g., CLIP, LLaVA) are trained on large-scale, lightly curated web datasets, leading them to learn unintended correlations between semantic concepts and unrelated visual signals. These associations degrade model accuracy by causing predictions to rely on incidental patterns rather than genuine visual understanding. Prior work has weaponized these correlations as an attack vector to manipulate model predictions, such as inserting a deceiving class text onto the image in a typographic attack. These attacks succeed due to VLMs' text-heavy bias-a result of captions that echo visible words rather than describing content. However, this attack has focused solely on text that matches the target class exactly, overlooking a broader range of correlations, including non-matching text and graphical symbols, which arise from the abundance of branding content in web-scale data. To address this gap, we introduce artifact-based attacks: a novel class of manipulations that mislead models using both non-matching text and graphical elements. Unlike typographic attacks, these artifacts are not predefined, making them harder to defend against but also more challenging to find. We address this by framing artifact attacks as a search problem and demonstrate their effectiveness across five datasets, with some artifacts reinforcing each other to reach 100% attack success rates. These attacks transfer across models with up to 90% effectiveness, making it possible to attack unseen models. To defend against these attacks, we extend prior work's artifact aware prompting to the graphical setting. We see a moderate reduction of success rates of up to 15% relative to standard prompts, suggesting a promising direction for enhancing model robustness.

摘要: 视觉语言模型(VLM)(例如，CLIP、LLaVA)在大规模、轻度精选的Web数据集上进行训练，导致它们学习语义概念和无关视觉信号之间的意外关联。这些关联导致预测依赖于附带模式而不是真正的视觉理解，从而降低了模型的准确性。以前的工作已经将这些相关性武器化为攻击矢量，以操纵模型预测，例如在排版攻击中将欺骗性类文本插入图像。这些攻击之所以成功，是因为VLMS的文本偏向--字幕呼应可见单词而不是描述内容的结果。然而，这种攻击只关注与目标类别完全匹配的文本，而忽略了更广泛的相关性，包括不匹配的文本和图形符号，这些相关性源于网络规模数据中丰富的品牌内容。为了弥补这一差距，我们引入了基于人工产物的攻击：一种使用不匹配文本和图形元素误导模型的新型操作。与排版攻击不同，这些文物不是预定义的，这使得它们更难防御，但也更难找到。我们通过将人工产物攻击作为一个搜索问题来解决这个问题，并在五个数据集上展示了它们的有效性，其中一些人工产物相互加强，达到100%的攻击成功率。这些攻击在模型之间转移，效率高达90%，使攻击看不见的模型成为可能。为了防御这些攻击，我们将以前工作的人工产物感知提示扩展到图形设置。我们看到，与标准提示相比，成功率适度降低，最高可达15%，这表明了增强模型稳健性的一个有希望的方向。



## **6. Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation**

助推器：通过减弱有害扰动来解决大型语言模型的有害微调 cs.CL

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2409.01586v4) [paper-pdf](http://arxiv.org/pdf/2409.01586v4)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Harmful fine-tuning attack poses serious safety concerns for large language models' fine-tuning-as-a-service. While existing defenses have been proposed to mitigate the issue, their performances are still far away from satisfactory, and the root cause of the problem has not been fully recovered. To this end, we in this paper show that harmful perturbation over the model weights could be a probable cause of alignment-broken. In order to attenuate the negative impact of harmful perturbation, we propose an alignment-stage solution, dubbed Booster. Technically, along with the original alignment loss, we append a loss regularizer in the alignment stage's optimization. The regularizer ensures that the model's harmful loss reduction after the simulated harmful perturbation is attenuated, thereby mitigating the subsequent fine-tuning risk. Empirical results show that Booster can effectively reduce the harmful score of the fine-tuned models while maintaining the performance of downstream tasks. Our code is available at https://github.com/git-disl/Booster.

摘要: 有害的微调攻击给大型语言模型的微调即服务带来了严重的安全问题。虽然已经提出了现有的防御措施来缓解这一问题，但它们的表现仍然远远不能令人满意，问题的根本原因尚未完全恢复。为此，我们在本文中表明，对模型权重的有害扰动可能是排列断裂的一个可能原因。为了减弱有害扰动的负面影响，我们提出了一种对准阶段的解决方案，称为Booster。在技术上，除了原始的对准损失外，我们还在对准阶段的优化中加入了损失正则化。正则化确保了模型在模拟的有害扰动后的有害损失减小，从而减轻了随后的微调风险。实验结果表明，Booster在保持下游任务性能的同时，能有效降低微调模型的有害分数。我们的代码可以在https://github.com/git-disl/Booster.上找到



## **7. A Framework to Assess Multilingual Vulnerabilities of LLMs**

评估法学硕士多语言脆弱性的框架 cs.CL

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13081v1) [paper-pdf](http://arxiv.org/pdf/2503.13081v1)

**Authors**: Likai Tang, Niruth Bogahawatta, Yasod Ginige, Jiarui Xu, Shixuan Sun, Surangika Ranathunga, Suranga Seneviratne

**Abstract**: Large Language Models (LLMs) are acquiring a wider range of capabilities, including understanding and responding in multiple languages. While they undergo safety training to prevent them from answering illegal questions, imbalances in training data and human evaluation resources can make these models more susceptible to attacks in low-resource languages (LRL). This paper proposes a framework to automatically assess the multilingual vulnerabilities of commonly used LLMs. Using our framework, we evaluated six LLMs across eight languages representing varying levels of resource availability. We validated the assessments generated by our automated framework through human evaluation in two languages, demonstrating that the framework's results align with human judgments in most cases. Our findings reveal vulnerabilities in LRL; however, these may pose minimal risk as they often stem from the model's poor performance, resulting in incoherent responses.

摘要: 大型语言模型（LLM）正在获得更广泛的能力，包括以多种语言理解和响应。虽然他们接受安全培训以防止他们回答非法问题，但训练数据和人力评估资源的不平衡可能使这些模型更容易受到低资源语言（LRL）的攻击。本文提出了一个自动评估常用LLM多语言漏洞的框架。使用我们的框架，我们评估了八种语言的六个LLM，代表不同级别的资源可用性。我们通过两种语言的人工评估验证了自动化框架生成的评估，证明框架的结果在大多数情况下与人类判断一致。我们的研究结果揭示了LRL中的漏洞;然而，这些漏洞可能构成的风险很小，因为它们通常源于模型的性能不佳，导致响应不一致。



## **8. TuBA: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning**

TuBA：具有指令调优的LLM后门攻击的跨语言可转移性 cs.CL

work in progress

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2404.19597v3) [paper-pdf](http://arxiv.org/pdf/2404.19597v3)

**Authors**: Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin I. P. Rubinstein, Trevor Cohn

**Abstract**: The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. Despite the increasing support for multilingual capabilities in open-source and proprietary LLMs, the impact of backdoor attacks on these systems remains largely under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data for one or two languages can affect the outputs for languages whose instruction-tuning data were not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5 and GPT-4o, with high attack success rates, surpassing 90% in more than 7 out of 12 languages across various scenarios. Our findings also indicate that more powerful models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments demonstrate 1) High Transferability: the backdoor mechanism operates successfully in cross-lingual response scenarios across 26 languages, achieving an average attack success rate of 99%, and 2) Robustness: the proposed attack remains effective even after defenses are applied. These findings expose critical security vulnerabilities in multilingual LLMs and highlight the urgent need for more robust, targeted defense strategies to address the unique challenges posed by cross-lingual backdoor transfer.

摘要: 后门攻击对以英语为中心的大型语言模型(LLM)的影响已被广泛研究-此类攻击可以通过在训练期间嵌入恶意行为来实现，并在触发恶意输出的特定条件下激活。尽管在开源和专有LLM中对多语言功能的支持越来越多，但后门攻击对这些系统的影响在很大程度上仍然没有得到充分的探索。我们的研究重点是针对多语言LLM的跨语言后门攻击，特别是调查毒化一到两种语言的指令调整数据如何影响那些指令调整数据没有中毒的语言的输出。尽管简单，但我们的经验分析表明，我们的方法在MT5和GPT-40等模型中表现出了显著的效果，在不同的场景下，在12种语言中的7种以上的攻击成功率超过90%。我们的发现还表明，更强大的模型显示出对可转移的跨语言后门攻击的易感性，这也适用于主要根据英语数据进行预训练的LLM，如Llama2、Llama3和Gema。此外，我们的实验表明：1)高可移植性：后门机制在26种语言的跨语言响应场景中成功运行，平均攻击成功率为99%；2)健壮性：所提出的攻击即使在实施防御后仍有效。这些发现暴露了多语言低成本管理中的关键安全漏洞，并突显了迫切需要更强大、更有针对性的防御战略，以应对跨语言后门转移带来的独特挑战。



## **9. How Good is my Histopathology Vision-Language Foundation Model? A Holistic Benchmark**

我的组织学视觉语言基础模型有多好？整体基准 eess.IV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12990v1) [paper-pdf](http://arxiv.org/pdf/2503.12990v1)

**Authors**: Roba Al Majzoub, Hashmat Malik, Muzammal Naseer, Zaigham Zaheer, Tariq Mahmood, Salman Khan, Fahad Khan

**Abstract**: Recently, histopathology vision-language foundation models (VLMs) have gained popularity due to their enhanced performance and generalizability across different downstream tasks. However, most existing histopathology benchmarks are either unimodal or limited in terms of diversity of clinical tasks, organs, and acquisition instruments, as well as their partial availability to the public due to patient data privacy. As a consequence, there is a lack of comprehensive evaluation of existing histopathology VLMs on a unified benchmark setting that better reflects a wide range of clinical scenarios. To address this gap, we introduce HistoVL, a fully open-source comprehensive benchmark comprising images acquired using up to 11 various acquisition tools that are paired with specifically crafted captions by incorporating class names and diverse pathology descriptions. Our Histo-VL includes 26 organs, 31 cancer types, and a wide variety of tissue obtained from 14 heterogeneous patient cohorts, totaling more than 5 million patches obtained from over 41K WSIs viewed under various magnification levels. We systematically evaluate existing histopathology VLMs on Histo-VL to simulate diverse tasks performed by experts in real-world clinical scenarios. Our analysis reveals interesting findings, including large sensitivity of most existing histopathology VLMs to textual changes with a drop in balanced accuracy of up to 25% in tasks such as Metastasis detection, low robustness to adversarial attacks, as well as improper calibration of models evident through high ECE values and low model prediction confidence, all of which can affect their clinical implementation.

摘要: 最近，组织病理学视觉-语言基础模型(VLM)因其在不同下游任务中增强的性能和普适性而广受欢迎。然而，现有的大多数组织病理学基准要么是单一的，要么在临床任务、器官和采集工具的多样性方面受到限制，而且由于患者数据的隐私，它们对公众的部分可用性。因此，在一个统一的基准设置上缺乏对现有组织病理学VLM的全面评估，以更好地反映广泛的临床情景。为了弥补这一差距，我们引入了HistoVL，这是一个完全开源的综合基准，包括使用多达11种不同的采集工具获取的图像，这些工具通过结合类名和不同的病理描述与专门制作的说明相匹配。我们的HISTO-VL包括26个器官，31种癌症类型，以及从14个不同的患者队列中获得的各种组织，总计超过500万个斑块，这些斑块是在不同放大水平下从超过41K的WSIS中获得的。我们系统地评估HISTO-VL上现有的组织病理学VLM，以模拟真实世界临床场景中专家执行的各种任务。我们的分析揭示了有趣的发现，包括大多数现有的组织病理学VLM对文本变化的高度敏感性，在转移检测等任务中平衡准确率下降高达25%，对对抗性攻击的稳健性低，以及通过高EC值和低模型预测置信度明显地对模型进行不正确的校准，所有这些都可能影响其临床实施。



## **10. MirrorGuard: Adaptive Defense Against Jailbreaks via Entropy-Guided Mirror Crafting**

收件箱警卫：通过熵引导镜子制作对越狱的自适应防御 cs.CR

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12931v1) [paper-pdf](http://arxiv.org/pdf/2503.12931v1)

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Litian Zhang, Lirong Qiu, Xi Zhang

**Abstract**: Defending large language models (LLMs) against jailbreak attacks is crucial for ensuring their safe deployment. Existing defense strategies generally rely on predefined static criteria to differentiate between harmful and benign prompts. However, such rigid rules are incapable of accommodating the inherent complexity and dynamic nature of real jailbreak attacks. In this paper, we propose a novel concept of ``mirror'' to enable dynamic and adaptive defense. A mirror refers to a dynamically generated prompt that mirrors the syntactic structure of the input while ensuring semantic safety. The personalized discrepancies between the input prompts and their corresponding mirrors serve as the guiding principles for defense. A new defense paradigm, MirrorGuard, is further proposed to detect and calibrate risky inputs based on such mirrors. An entropy-based detection metric, Relative Input Uncertainty (RIU), is integrated into MirrorGuard to quantify the discrepancies between input prompts and mirrors. MirrorGuard is evaluated on several popular datasets, demonstrating state-of-the-art defense performance while maintaining general effectiveness.

摘要: 保护大型语言模型(LLM)免受越狱攻击对于确保它们的安全部署至关重要。现有的防御策略通常依赖于预定义的静态标准来区分有害提示和良性提示。然而，这种僵化的规则无法适应真正越狱攻击的内在复杂性和动态性质。本文提出了一种新的“镜像”概念，以实现动态和自适应的防御。镜像是指动态生成的提示，它反映了输入的句法结构，同时确保了语义安全。输入提示与其对应的镜像之间的个性化差异是答辩的指导原则。此外，还提出了一种新的防御范例MirrorGuard，用于检测和校准基于此类镜像的危险输入。基于熵的检测指标相对输入不确定性(RIU)被集成到MirrorGuard中，以量化输入提示和镜像之间的差异。MirrorGuard在几个流行的数据集上进行了评估，在保持总体有效性的同时展示了最先进的防御性能。



## **11. When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers**

当推理中的“能力”打开脆弱之门：通过新颖复杂密码越狱LLM cs.CL

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2402.10601v3) [paper-pdf](http://arxiv.org/pdf/2402.10601v3)

**Authors**: Divij Handa, Zehua Zhang, Amir Saeidi, Shrinidhi Kumbhar, Chitta Baral

**Abstract**: Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to these jailbreaking attacks, with success rates on GPT-4o escalating from 40% under ACE to 78% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.

摘要: 大型语言模型(LLM)安全方面的最新进展主要集中在缓解使用自然语言或常见密码(例如Base64)编写的攻击，这些攻击可能会集成到较新模型的安全培训中。然而，我们揭示了一个矛盾的弱点：随着LLM在推理方面的进步，它们在不经意间变得更容易受到新的越狱攻击。增强的推理使LLM能够解释复杂的指令和解码复杂的用户定义的密码，从而产生了一个可利用的安全漏洞。为了研究此漏洞，我们引入了使用自定义加密(ACE)的攻击，这是一种使用新密码对恶意查询进行编码的越狱技术。对ACE进行了扩展，引入了使用自定义加密的分层攻击(LACE)，它采用多层密码来放大攻击的复杂性。此外，我们还开发了用于评估LLMS在解密加密良性文本时的准确性的基准测试CipherBch。我们的实验揭示了一个关键的权衡：更有能力解码密码的LLM更容易受到这些越狱攻击，GPT-40上的成功率从ACE下的40%上升到LACE下的78%。这些发现突显了一个重要的洞察：随着LLM变得更加擅长破译复杂的用户密码--其中许多密码无法先发制人地包括在安全培训中--它们变得越来越容易被利用。



## **12. h4rm3l: A language for Composable Jailbreak Attack Synthesis**

h4 rm3l：可组合越狱攻击合成语言 cs.CR

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2408.04811v3) [paper-pdf](http://arxiv.org/pdf/2408.04811v3)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: Despite their demonstrated valuable capabilities, state-of-the-art (SOTA) widely deployed large language models (LLMs) still have the potential to cause harm to society due to the ineffectiveness of their safety filters, which can be bypassed by prompt transformations called jailbreak attacks. Current approaches to LLM safety assessment, which employ datasets of templated prompts and benchmarking pipelines, fail to cover sufficiently large and diverse sets of jailbreak attacks, leading to the widespread deployment of unsafe LLMs. Recent research showed that novel jailbreak attacks could be derived by composition; however, a formal composable representation for jailbreak attacks, which, among other benefits, could enable the exploration of a large compositional space of jailbreak attacks through program synthesis methods, has not been previously proposed. We introduce h4rm3l, a novel approach that addresses this gap with a human-readable domain-specific language (DSL). Our framework comprises: (1) The h4rm3l DSL, which formally expresses jailbreak attacks as compositions of parameterized string transformation primitives. (2) A synthesizer with bandit algorithms that efficiently generates jailbreak attacks optimized for a target black box LLM. (3) The h4rm3l red-teaming software toolkit that employs the previous two components and an automated harmful LLM behavior classifier that is strongly aligned with human judgment. We demonstrate h4rm3l's efficacy by synthesizing a dataset of 2656 successful novel jailbreak attacks targeting 6 SOTA open-source and proprietary LLMs, and by benchmarking those models against a subset of these synthesized attacks. Our results show that h4rm3l's synthesized attacks are diverse and more successful than existing jailbreak attacks in literature, with success rates exceeding 90% on SOTA LLMs.

摘要: 尽管被广泛部署的最先进的大型语言模型(SOTA)显示了其宝贵的能力，但由于其安全过滤器的无效，仍有可能对社会造成危害，这可以通过称为越狱攻击的快速转换来绕过。目前的LLM安全评估方法使用模板化提示和基准管道的数据集，无法覆盖足够大和多样化的越狱攻击集，导致不安全的LLM被广泛部署。最近的研究表明，新的越狱攻击可以通过组合来派生；然而，以前还没有提出用于越狱攻击的形式可组合表示，除了其他优点之外，它可以通过程序合成方法来探索越狱攻击的大组成空间。我们介绍了h4rm3l，这是一种用人类可读的领域特定语言(DSL)来解决这一差距的新方法。我们的框架包括：(1)h4rm3l DSL，它将越狱攻击形式化地表达为参数化串转换原语的组合。(2)采用强盗算法的合成器，能够高效地生成针对目标黑盒LLM进行优化的越狱攻击。(3)使用前两个组件的h4rm3l红队软件工具包，以及与人类判断高度一致的自动有害LLM行为分类器。我们通过合成针对6个Sota开源和专有LLM的2656个成功的新型越狱攻击的数据集，并将这些模型与这些合成攻击的子集进行基准比较，展示了h4rm3l的有效性。我们的结果表明，h4rm3l的综合攻击是多样的，而且比文献中现有的越狱攻击更成功，在Sota LLMS上的成功率超过90%。



## **13. Systematic Categorization, Construction and Evaluation of New Attacks against Multi-modal Mobile GUI Agents**

针对多模式移动图形用户界面代理的新攻击的系统分类、构建和评估 cs.CR

Preprint. Work in progress

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2407.09295v3) [paper-pdf](http://arxiv.org/pdf/2407.09295v3)

**Authors**: Yulong Yang, Xinshan Yang, Shuaidong Li, Chenhao Lin, Zhengyu Zhao, Chao Shen, Tianwei Zhang

**Abstract**: The integration of Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs) into mobile GUI agents has significantly enhanced user efficiency and experience. However, this advancement also introduces potential security vulnerabilities that have yet to be thoroughly explored. In this paper, we present a systematic security investigation of multi-modal mobile GUI agents, addressing this critical gap in the existing literature. Our contributions are twofold: (1) we propose a novel threat modeling methodology, leading to the discovery and feasibility analysis of 34 previously unreported attacks, and (2) we design an attack framework to systematically construct and evaluate these threats. Through a combination of real-world case studies and extensive dataset-driven experiments, we validate the severity and practicality of those attacks, highlighting the pressing need for robust security measures in mobile GUI systems.

摘要: 将大型语言模型（LLM）和多模式大型语言模型（MLLM）集成到移动图形用户界面代理中，显着提高了用户效率和体验。然而，这一进步也引入了尚未彻底探索的潜在安全漏洞。在本文中，我们对多模式移动图形用户界面代理进行了系统性的安全研究，以解决现有文献中的这一关键空白。我们的贡献有双重：（1）我们提出了一种新颖的威胁建模方法，发现了34种之前未报告的攻击并进行了可行性分析，（2）我们设计了一个攻击框架来系统地构建和评估这些威胁。通过现实世界的案例研究和广泛的厕所驱动实验的结合，我们验证了这些攻击的严重性和实用性，强调了移动图形用户界面系统中对强大安全措施的迫切需求。



## **14. One Goal, Many Challenges: Robust Preference Optimization Amid Content-Aware and Multi-Source Noise**

一个目标，诸多挑战：内容感知和多源噪音中的稳健偏好优化 cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12301v1) [paper-pdf](http://arxiv.org/pdf/2503.12301v1)

**Authors**: Amirabbas Afzali, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi, Sanjay Lall

**Abstract**: Large Language Models (LLMs) have made significant strides in generating human-like responses, largely due to preference alignment techniques. However, these methods often assume unbiased human feedback, which is rarely the case in real-world scenarios. This paper introduces Content-Aware Noise-Resilient Preference Optimization (CNRPO), a novel framework that addresses multiple sources of content-dependent noise in preference learning. CNRPO employs a multi-objective optimization approach to separate true preferences from content-aware noises, effectively mitigating their impact. We leverage backdoor attack mechanisms to efficiently learn and control various noise sources within a single model. Theoretical analysis and extensive experiments on different synthetic noisy datasets demonstrate that CNRPO significantly improves alignment with primary human preferences while controlling for secondary noises and biases, such as response length and harmfulness.

摘要: 大型语言模型（LLM）在生成类人响应方面取得了重大进展，这主要归功于偏好对齐技术。然而，这些方法通常假设无偏见的人类反馈，而在现实世界场景中情况很少。本文介绍了内容感知噪音弹性偏好优化（CNRPO），这是一种新型框架，可解决偏好学习中多个内容相关噪音来源。CNRPO采用多目标优化方法将真实偏好与内容感知噪音分开，有效减轻其影响。我们利用后门攻击机制来有效地学习和控制单个模型内的各种噪音源。对不同合成噪音数据集的理论分析和广泛实验表明，CNRPO显着改善了与人类主要偏好的一致性，同时控制次要噪音和偏差，例如响应长度和危害性。



## **15. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2407.20361v3) [paper-pdf](http://arxiv.org/pdf/2407.20361v3)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of three existing task-specific models -- Stack model, VisualPhishNet, and Phishpedia -- against PhishOracle-generated adversarial phishing webpages and observe a significant drop in their detection rates. In contrast, a multimodal large language model (MLLM)-based phishing detector demonstrates stronger robustness against these adversarial attacks but still is prone to evasion. Our findings highlight the vulnerability of phishing detection models to adversarial attacks, emphasizing the need for more robust detection approaches. Furthermore, we conduct a user study to evaluate whether PhishOracle-generated adversarial phishing webpages can deceive users. The results show that many of these phishing webpages evade not only existing detection models but also users. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources will be made publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的三种特定任务的模型--Stack模型、VisualPhishNet和Phishpedia--对PhishOracle生成的敌意钓鱼网页的稳健性，并观察到它们的检测率显著下降。相比之下，基于多模式大语言模型(MLLM)的网络钓鱼检测器对这些敌意攻击表现出更强的稳健性，但仍然容易被规避。我们的发现突出了网络钓鱼检测模型对对手攻击的脆弱性，强调了需要更强大的检测方法。此外，我们还进行了一项用户研究，以评估PhishOracle生成的敌意钓鱼网页是否可以欺骗用户。结果表明，许多钓鱼网页不仅规避了现有的检测模型，而且还规避了用户。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都将在GitHub上公开提供。



## **16. JailGuard: A Universal Detection Framework for LLM Prompt-based Attacks**

JailGuard：针对LLM基于预算的攻击的通用检测框架 cs.CR

40 pages, 12 figures

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2312.10766v4) [paper-pdf](http://arxiv.org/pdf/2312.10766v4)

**Authors**: Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Ming Hu, Jie Zhang, Yang Liu, Shiqing Ma, Chao Shen

**Abstract**: The systems and software powered by Large Language Models (LLMs) and Multi-Modal LLMs (MLLMs) have played a critical role in numerous scenarios. However, current LLM systems are vulnerable to prompt-based attacks, with jailbreaking attacks enabling the LLM system to generate harmful content, while hijacking attacks manipulate the LLM system to perform attacker-desired tasks, underscoring the necessity for detection tools. Unfortunately, existing detecting approaches are usually tailored to specific attacks, resulting in poor generalization in detecting various attacks across different modalities. To address it, we propose JailGuard, a universal detection framework deployed on top of LLM systems for prompt-based attacks across text and image modalities. JailGuard operates on the principle that attacks are inherently less robust than benign ones. Specifically, JailGuard mutates untrusted inputs to generate variants and leverages the discrepancy of the variants' responses on the target model to distinguish attack samples from benign samples. We implement 18 mutators for text and image inputs and design a mutator combination policy to further improve detection generalization. The evaluation on the dataset containing 15 known attack types suggests that JailGuard achieves the best detection accuracy of 86.14%/82.90% on text and image inputs, outperforming state-of-the-art methods by 11.81%-25.73% and 12.20%-21.40%.

摘要: 由大型语言模型(LLM)和多模式LLM(MLLMS)支持的系统和软件在许多情况下发挥了关键作用。然而，当前的LLM系统容易受到基于提示的攻击，越狱攻击使LLM系统能够生成有害内容，而劫持攻击操纵LLM系统执行攻击者想要的任务，这突显了检测工具的必要性。遗憾的是，现有的检测方法通常是针对特定的攻击量身定做的，导致在检测不同模式的各种攻击时通用性较差。为了解决这个问题，我们提出了JailGuard，这是一个部署在LLM系统之上的通用检测框架，用于跨文本和图像通道的基于提示的攻击。JailGuard的运作原则是，攻击天生就不如良性攻击那么强大。具体地说，JailGuard会变异不可信的输入以生成变体，并利用变体对目标模型的响应差异来区分攻击样本和良性样本。我们为文本和图像输入实现了18个变异器，并设计了变异器组合策略，进一步提高了检测的泛化能力。对包含15种已知攻击类型的数据集的评估表明，JailGuard在文本和图像输入上的最佳检测准确率为86.14%/82.90%，分别比最先进的方法高11.81%-25.73%和12.20%-21.40%。



## **17. Making Every Step Effective: Jailbreaking Large Vision-Language Models Through Hierarchical KV Equalization**

让每一步都有效：通过分层KN均衡化破解大型视觉语言模型 cs.CV

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11750v1) [paper-pdf](http://arxiv.org/pdf/2503.11750v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Jun Liu, Muhao Chen, Zi Huang, Yujun Cai

**Abstract**: In the realm of large vision-language models (LVLMs), adversarial jailbreak attacks serve as a red-teaming approach to identify safety vulnerabilities of these models and their associated defense mechanisms. However, we identify a critical limitation: not every adversarial optimization step leads to a positive outcome, and indiscriminately accepting optimization results at each step may reduce the overall attack success rate. To address this challenge, we introduce HKVE (Hierarchical Key-Value Equalization), an innovative jailbreaking framework that selectively accepts gradient optimization results based on the distribution of attention scores across different layers, ensuring that every optimization step positively contributes to the attack. Extensive experiments demonstrate HKVE's significant effectiveness, achieving attack success rates of 75.08% on MiniGPT4, 85.84% on LLaVA and 81.00% on Qwen-VL, substantially outperforming existing methods by margins of 20.43\%, 21.01\% and 26.43\% respectively. Furthermore, making every step effective not only leads to an increase in attack success rate but also allows for a reduction in the number of iterations, thereby lowering computational costs. Warning: This paper contains potentially harmful example data.

摘要: 在大型视觉语言模型领域，对抗性越狱攻击是一种识别这些模型及其相关防御机制的安全漏洞的红团队方法。然而，我们发现了一个关键的限制：并不是每个对抗性的优化步骤都会带来积极的结果，并且不分青红皂白地接受每个步骤的优化结果可能会降低总体攻击成功率。为了应对这一挑战，我们引入了HKVE(分层键值均衡)，这是一个创新的越狱框架，它根据注意力分数在不同层的分布来选择性地接受梯度优化结果，确保每一个优化步骤都对攻击有积极的贡献。大量实验表明，HKVE具有显著的攻击效果，在MiniGPT4、LLaVA和Qwen-VL上的攻击成功率分别达到75.08%、85.84%和81.00%，分别比现有方法高出20.43、21.01和26.43%。此外，使每一步都有效，不仅可以提高攻击成功率，还可以减少迭代次数，从而降低计算成本。警告：本文包含可能有害的示例数据。



## **18. Tit-for-Tat: Safeguarding Large Vision-Language Models Against Jailbreak Attacks via Adversarial Defense**

针锋相对：通过对抗性防御保护大型视觉语言模型免受越狱攻击 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11619v1) [paper-pdf](http://arxiv.org/pdf/2503.11619v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Ming-Hsuan Yang, Jun Liu, Chengcheng Tang, Zi Huang, Yujun Cai

**Abstract**: Deploying large vision-language models (LVLMs) introduces a unique vulnerability: susceptibility to malicious attacks via visual inputs. However, existing defense methods suffer from two key limitations: (1) They solely focus on textual defenses, fail to directly address threats in the visual domain where attacks originate, and (2) the additional processing steps often incur significant computational overhead or compromise model performance on benign tasks. Building on these insights, we propose ESIII (Embedding Security Instructions Into Images), a novel methodology for transforming the visual space from a source of vulnerability into an active defense mechanism. Initially, we embed security instructions into defensive images through gradient-based optimization, obtaining security instructions in the visual dimension. Subsequently, we integrate security instructions from visual and textual dimensions with the input query. The collaboration between security instructions from different dimensions ensures comprehensive security protection. Extensive experiments demonstrate that our approach effectively fortifies the robustness of LVLMs against such attacks while preserving their performance on standard benign tasks and incurring an imperceptible increase in time costs.

摘要: 部署大型视觉语言模型(LVLM)引入了一个独特的漏洞：通过视觉输入易受恶意攻击。然而，现有的防御方法有两个关键的局限性：(1)它们只关注文本防御，不能直接应对发起攻击的视域中的威胁；(2)额外的处理步骤往往会导致巨大的计算开销或对良性任务的模型性能造成影响。基于这些见解，我们提出了ESIII(Embedding Security Instructions To Images)，这是一种将视觉空间从易受攻击源转变为主动防御机制的新方法。首先，通过基于梯度的优化将安全指令嵌入到防御图像中，得到视觉维度的安全指令。随后，我们将来自视觉和文本维度的安全指令与输入查询集成在一起。来自不同维度的安全指令之间的协作确保了全面的安全防护。大量的实验表明，我们的方法有效地增强了LVLM对此类攻击的健壮性，同时保持了它们在标准良性任务上的性能，并导致了不可察觉的时间开销的增加。



## **19. Align in Depth: Defending Jailbreak Attacks via Progressive Answer Detoxification**

深度结盟：通过渐进式答案去规范化捍卫越狱袭击 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11185v1) [paper-pdf](http://arxiv.org/pdf/2503.11185v1)

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks, which use crafted prompts to elicit toxic responses. These attacks exploit LLMs' difficulty in dynamically detecting harmful intents during the generation process. Traditional safety alignment methods, often relying on the initial few generation steps, are ineffective due to limited computational budget. This paper proposes DEEPALIGN, a robust defense framework that fine-tunes LLMs to progressively detoxify generated content, significantly improving both the computational budget and effectiveness of mitigating harmful generation. Our approach uses a hybrid loss function operating on hidden states to directly improve LLMs' inherent awareness of toxity during generation. Furthermore, we redefine safe responses by generating semantically relevant answers to harmful queries, thereby increasing robustness against representation-mutation attacks. Evaluations across multiple LLMs demonstrate state-of-the-art defense performance against six different attack types, reducing Attack Success Rates by up to two orders of magnitude compared to previous state-of-the-art defense while preserving utility. This work advances LLM safety by addressing limitations of conventional alignment through dynamic, context-aware mitigation.

摘要: 大型语言模型(LLM)容易受到越狱攻击，这些攻击使用精心编制的提示来引发有毒响应。这些攻击利用了LLMS在生成过程中动态检测有害意图的困难。传统的安全配准方法往往依赖于初始的几个生成步骤，由于计算预算有限，效率不高。本文提出了DEEPALIGN，这是一个健壮的防御框架，它对LLMS进行微调，以逐步对生成的内容进行解毒，显著提高了计算预算和减轻有害生成的有效性。我们的方法使用一种对隐态进行操作的混合损失函数来直接提高LLMS在生成过程中对毒性的固有意识。此外，我们通过生成对有害查询的语义相关答案来重新定义安全响应，从而提高了对表示-突变攻击的健壮性。对多个LLM的评估显示了针对六种不同攻击类型的最先进的防御性能，与以前最先进的防御相比，攻击成功率降低了高达两个数量级，同时保留了实用性。这项工作通过动态、上下文感知的缓解来解决传统对准的局限性，从而提高了LLM的安全性。



## **20. Towards a Systematic Evaluation of Hallucinations in Large-Vision Language Models**

对大视野语言模型中的幻觉进行系统评估 cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2412.20622v2) [paper-pdf](http://arxiv.org/pdf/2412.20622v2)

**Authors**: Ashish Seth, Dinesh Manocha, Chirag Agarwal

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in complex multimodal tasks. However, these models still suffer from hallucinations, particularly when required to implicitly recognize or infer diverse visual entities from images for complex vision-language tasks. To address this challenge, we propose HALLUCINOGEN, a novel visual question answering (VQA) benchmark that employs contextual reasoning prompts as hallucination attacks to evaluate the extent of hallucination in state-of-the-art LVLMs. Our benchmark provides a comprehensive study of the implicit reasoning capabilities of these models by first categorizing visual entities based on the ease of recognition in an image as either salient (prominent, visibly recognizable objects such as a car) or latent entities (such as identifying a disease from a chest X-ray), which are not readily visible and require domain knowledge or contextual reasoning for accurate inference. Next, we design hallucination attacks for both types of entities to assess hallucinations in LVLMs while performing various vision-language tasks, such as locating or reasoning about specific entities within an image, where models must perform implicit reasoning by verifying the existence of the queried entity within the image before generating responses. Finally, our extensive evaluations of eleven LVLMs, including powerful open-source models (like LLaMA-3.2 and DeepSeek-V2), commercial models like Gemini, and two hallucination mitigation strategies across multiple datasets, demonstrate that current LVLMs remain susceptible to hallucination attacks.

摘要: 大型视觉语言模型在复杂的多通道任务中表现出显著的性能。然而，这些模型仍然存在幻觉，特别是在复杂的视觉语言任务中需要从图像中隐含地识别或推断不同的视觉实体时。为了应对这一挑战，我们提出了幻觉剂，这是一种新颖的视觉问答基准，它使用上下文推理提示作为幻觉攻击来评估最先进的LVLM中的幻觉程度。我们的基准提供了对这些模型的隐含推理能力的全面研究，首先根据图像中识别的容易程度将视觉实体分类为突出实体(突出的、可视识别的对象，如汽车)或潜在实体(如从胸部X光识别疾病)，这些实体不容易看到，需要领域知识或上下文推理才能进行准确的推理。接下来，我们为两种类型的实体设计幻觉攻击，以评估LVLMS中的幻觉，同时执行各种视觉语言任务，如定位或关于图像中的特定实体进行推理，其中模型必须在生成响应之前通过验证被查询实体在图像中的存在来执行隐式推理。最后，我们对11个LVLM进行了广泛的评估，包括强大的开源模型(如Llama-3.2和DeepSeek-V2)、商业模型(如Gemini)，以及跨多个数据集的两种幻觉缓解策略，表明当前的LVLM仍然容易受到幻觉攻击。



## **21. ChatGPT Encounters Morphing Attack Detection: Zero-Shot MAD with Multi-Modal Large Language Models and General Vision Models**

ChatGPT遭遇变形攻击检测：具有多模式大语言模型和通用视觉模型的零镜头MAD cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10937v1) [paper-pdf](http://arxiv.org/pdf/2503.10937v1)

**Authors**: Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch

**Abstract**: Face Recognition Systems (FRS) are increasingly vulnerable to face-morphing attacks, prompting the development of Morphing Attack Detection (MAD) algorithms. However, a key challenge in MAD lies in its limited generalizability to unseen data and its lack of explainability-critical for practical application environments such as enrolment stations and automated border control systems. Recognizing that most existing MAD algorithms rely on supervised learning paradigms, this work explores a novel approach to MAD using zero-shot learning leveraged on Large Language Models (LLMs). We propose two types of zero-shot MAD algorithms: one leveraging general vision models and the other utilizing multimodal LLMs. For general vision models, we address the MAD task by computing the mean support embedding of an independent support set without using morphed images. For the LLM-based approach, we employ the state-of-the-art GPT-4 Turbo API with carefully crafted prompts. To evaluate the feasibility of zero-shot MAD and the effectiveness of the proposed methods, we constructed a print-scan morph dataset featuring various unseen morphing algorithms, simulating challenging real-world application scenarios. Experimental results demonstrated notable detection accuracy, validating the applicability of zero-shot learning for MAD tasks. Additionally, our investigation into LLM-based MAD revealed that multimodal LLMs, such as ChatGPT, exhibit remarkable generalizability to untrained MAD tasks. Furthermore, they possess a unique ability to provide explanations and guidance, which can enhance transparency and usability for end-users in practical applications.

摘要: 人脸识别系统(FRS)越来越容易受到变形攻击，这促使变形攻击检测(MAD)算法的发展。然而，MAD的一个关键挑战在于，它对看不见的数据的概括性有限，而且缺乏可解释性--这对于招生站和自动边界控制系统等实际应用环境至关重要。认识到大多数现有的MAD算法依赖于有监督的学习范例，该工作探索了一种利用大型语言模型(LLMS)的零镜头学习来实现MAD的新方法。我们提出了两种类型的零镜头MAD算法：一种利用一般的视觉模型，另一种利用多模式LLMS。对于一般的视觉模型，我们通过计算独立支持集的平均支持嵌入来解决MAD任务，而不使用变形图像。对于基于LLM的方法，我们使用了最先进的GPT-4 Turbo API和精心制作的提示。为了评估零镜头MAD的可行性和提出的方法的有效性，我们构建了一个包含各种不可见变形算法的打印扫描变形数据集，模拟了具有挑战性的真实应用场景。实验结果显示了显著的检测精度，验证了零镜头学习在MAD任务中的适用性。此外，我们对基于LLM的MAD的研究表明，多通道LLM，如ChatGPT，对未经训练的MAD任务表现出显著的泛化能力。此外，它们还具有提供解释和指导的独特能力，这可以在实际应用中提高最终用户的透明度和可用性。



## **22. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

令人沮丧的简单但高效的攻击基线：针对GPT-4.5/4 o/o 1的强黑匣子模型的成功率超过90% cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.

摘要: 尽管在开源的大型视觉语言模型(LVLM)上性能很好，但基于传输的定向攻击对黑盒商业LVLM往往失败。分析失败的对抗性扰动发现，学习的扰动通常源于均匀分布，缺乏明确的语义细节，导致意外响应。这种严重的语义信息缺失导致商业LVLM要么完全忽略该扰动，要么曲解其嵌入的语义，从而导致攻击失败。为了克服这些问题，我们注意到，识别核心语义对象是用各种数据集和方法训练的模型的关键目标。这种洞察力激发了我们的方法，通过在局部区域内编码显式的语义细节来细化语义清晰度，从而确保互操作性和捕获更细粒度的特征，并通过将修改集中在语义丰富的区域而不是统一地应用它们。为了实现这一点，我们提出了一种简单而高效的解决方案：在每个优化步骤中，通过控制纵横比和比例来随机裁剪敌对图像，调整大小，然后在嵌入空间中与目标图像对齐。实验结果证实了我们的假设。我们的对抗性例子使用聚焦于关键区域的局部聚集扰动来制作，表现出出奇地好的可转换性，可以用于商业LVLM，包括GPT-4.5、GPT-40、Gemini-2.0-Flash、Claude-3.5-十四行诗、Claude-3.7-十四行诗，甚至像o1、Claude-3.7-Think和Gemini-2.0-Flash-Think这样的推理模型。我们的方法在GPT-4.5、40o和o1上的成功率超过90%，大大超过了之前所有最先进的攻击方法。我们在不同配置和训练代码下的优化对抗性示例可在https://github.com/VILA-Lab/M-Attack.获得



## **23. ASIDE: Architectural Separation of Instructions and Data in Language Models**

ASIDE：语言模型中指令和数据的架构分离 cs.LG

ICLR 2025 Workshop on Building Trust in Language Models and  Applications

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10566v1) [paper-pdf](http://arxiv.org/pdf/2503.10566v1)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Soroush Tabesh, Alexandra Volkova, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, and this makes them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause for the success of prompt injection attacks. In this work, we propose an architectural change, ASIDE, that allows the model to clearly separate between instructions and data by using separate embeddings for them. Instead of training the embeddings from scratch, we propose a method to convert an existing model to ASIDE form by using two copies of the original model's embeddings layer, and applying an orthogonal rotation to one of them. We demonstrate the effectiveness of our method by showing (1) highly increased instruction-data separation scores without a loss in model capabilities and (2) competitive results on prompt injection benchmarks, even without dedicated safety training. Additionally, we study the working mechanism behind our method through an analysis of model representations.

摘要: 尽管性能卓越，但大型语言模型缺乏基本的安全功能，这使得它们容易受到大量恶意攻击。特别是，以前的工作已经确定指令和数据之间没有内在的分离是快速注入攻击成功的根本原因。在这项工作中，我们提出了一种架构改变，允许模型通过对指令和数据使用单独的嵌入来明确地分离它们。我们没有从头开始训练嵌入，而是提出了一种方法，通过使用原始模型的嵌入层的两个副本并对其中一个模型应用正交旋转来将现有模型转换为旁注形式。我们通过展示(1)在不损失模型能力的情况下极大地提高指令-数据分离分数和(2)即使在没有专门的安全培训的情况下也在快速注入基准上具有竞争力的结果来证明我们方法的有效性。此外，我们还通过对模型表示的分析，研究了该方法背后的工作机制。



## **24. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

TH-Bench：通过机器生成文本检测器上人性化人工智能文本来评估逃避攻击 cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.

摘要: 随着大型语言模型(LLM)的发展，机器生成文本(MGTS)变得越来越流畅、高质量和信息丰富。现有的大范围MGT探测器旨在识别MGT，以防止抄袭和错误信息的传播。然而，攻击者试图使MGTS人性化以躲避检测(称为躲避攻击)，这只需要进行少量修改即可绕过MGT检测器。遗憾的是，现有的攻击通常缺乏统一和全面的评估框架，因为它们是使用不同的实验设置、模型架构和数据集进行评估的。为了填补这一空白，我们引入了文本人性化基准(TH-BENCH)，这是第一个评估针对MGT检测器的躲避攻击的综合基准。TH-BENCH从三个关键维度对攻击进行评估：规避效率、文本质量和计算开销。我们的广泛实验评估了6种针对6个数据集的13个MGT检测器的最先进攻击，这些攻击跨越19个域，由11个广泛使用的LLM生成。我们的发现表明，没有一次逃避攻击在所有三个维度上都表现出色。通过深入的分析，我们突出了不同攻击的优势和局限性。更重要的是，我们确定了三个维度之间的权衡，并提出了两个优化见解。通过初步实验，验证了它们的正确性和有效性，为以后的研究提供了潜在的方向。



## **25. Prompt Inversion Attack against Collaborative Inference of Large Language Models**

针对大型语言模型协作推理的提示倒置攻击 cs.CR

To appear at IEEE Symposium on Security and Privacy 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09022v2) [paper-pdf](http://arxiv.org/pdf/2503.09022v2)

**Authors**: Wenjie Qu, Yuguang Zhou, Yongji Wu, Tingsong Xiao, Binhang Yuan, Yiming Li, Jiaheng Zhang

**Abstract**: Large language models (LLMs) have been widely applied for their remarkable capability of content generation. However, the practical use of open-source LLMs is hindered by high resource requirements, making deployment expensive and limiting widespread development. The collaborative inference is a promising solution for this problem, in which users collaborate by each hosting a subset of layers and transmitting intermediate activation. Many companies are building collaborative inference platforms to reduce LLM serving costs, leveraging users' underutilized GPUs. Despite widespread interest in collaborative inference within academia and industry, the privacy risks associated with LLM collaborative inference have not been well studied. This is largely because of the challenge posed by inverting LLM activation due to its strong non-linearity.   In this paper, to validate the severity of privacy threats in LLM collaborative inference, we introduce the concept of prompt inversion attack (PIA), where a malicious participant intends to recover the input prompt through the activation transmitted by its previous participant. Extensive experiments show that our PIA method substantially outperforms existing baselines. For example, our method achieves an 88.4\% token accuracy on the Skytrax dataset with the Llama-65B model when inverting the maximum number of transformer layers, while the best baseline method only achieves 22.8\% accuracy. The results verify the effectiveness of our PIA attack and highlights its practical threat to LLM collaborative inference systems.

摘要: 大语言模型以其卓越的内容生成能力得到了广泛的应用。然而，开源LLMS的实际使用受到高资源要求的阻碍，使得部署成本高昂，并限制了广泛的发展。协作推理是解决这一问题的一种很有前途的解决方案，其中每个用户通过托管一个层的子集并传递中间激活来进行协作。许多公司正在构建协作推理平台，以利用用户未充分利用的GPU来降低LLM服务成本。尽管学术界和工业界对协同推理产生了广泛的兴趣，但与LLM协同推理相关的隐私风险尚未得到很好的研究。这在很大程度上是因为LLM激活的反转带来了挑战，因为它具有很强的非线性。为了验证LLM协同推理中隐私威胁的严重性，我们引入了即时反转攻击(PIA)的概念，恶意参与者试图通过其先前参与者发送的激活来恢复输入提示。广泛的实验表明，我们的PIA方法的性能大大优于现有的基线。例如，我们的方法在使用Llama-65B模型的Skytrax数据集上，在反演最大变压器层数时达到了88.4%的令牌精度，而最佳基线方法只达到了22.8%的精度。实验结果验证了PIA攻击的有效性，并突出了其对LLM协同推理系统的实际威胁。



## **26. ExtremeAIGC: Benchmarking LMM Vulnerability to AI-Generated Extremist Content**

ExtremeAIGC：LMM漏洞针对人工智能生成的极端主义内容进行基准测试 cs.CR

Preprint

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09964v1) [paper-pdf](http://arxiv.org/pdf/2503.09964v1)

**Authors**: Bhavik Chandna, Mariam Aboujenane, Usman Naseem

**Abstract**: Large Multimodal Models (LMMs) are increasingly vulnerable to AI-generated extremist content, including photorealistic images and text, which can be used to bypass safety mechanisms and generate harmful outputs. However, existing datasets for evaluating LMM robustness offer limited exploration of extremist content, often lacking AI-generated images, diverse image generation models, and comprehensive coverage of historical events, which hinders a complete assessment of model vulnerabilities. To fill this gap, we introduce ExtremeAIGC, a benchmark dataset and evaluation framework designed to assess LMM vulnerabilities against such content. ExtremeAIGC simulates real-world events and malicious use cases by curating diverse text- and image-based examples crafted using state-of-the-art image generation techniques. Our study reveals alarming weaknesses in LMMs, demonstrating that even cutting-edge safety measures fail to prevent the generation of extremist material. We systematically quantify the success rates of various attack strategies, exposing critical gaps in current defenses and emphasizing the need for more robust mitigation strategies.

摘要: 大型多模式模型(LMM)越来越容易受到人工智能生成的极端主义内容的影响，包括照片级真实感图像和文本，这些内容可用于绕过安全机制并产生有害输出。然而，现有的用于评估LMM稳健性的数据集提供了对极端主义内容的有限探索，往往缺乏人工智能生成的图像、多样化的图像生成模型以及对历史事件的全面覆盖，这阻碍了对模型漏洞的完整评估。为了填补这一空白，我们引入了ExtremeAIGC，这是一个基准数据集和评估框架，旨在评估针对此类内容的LMM漏洞。ExtremeAIGC通过策划使用最先进的图像生成技术制作的各种基于文本和图像的示例来模拟真实世界的事件和恶意使用案例。我们的研究揭示了LMM令人震惊的弱点，表明即使是尖端的安全措施也无法防止极端主义材料的产生。我们系统地量化了各种攻击策略的成功率，暴露了当前防御中的关键差距，并强调了需要更强大的缓解策略。



## **27. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

AISTATS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2412.08099v4) [paper-pdf](http://arxiv.org/pdf/2412.08099v4)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like LLMTime with GPT-3.5, GPT-4, LLaMa, and Mistral, TimeGPT, and TimeLLM show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications. The code repository can be found at https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.

摘要: 大型语言模型(LLM)最近在时间序列预测方面显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验包括LLMTime与GPT-3.5、GPT-4、Llama和Mistral、TimeGPT和TimeLLM的模型，实验表明，对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。代码存储库可在https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.上找到



## **28. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

CyberLLMDirecct：使用网络安全数据分析精调LLM安全性的新数据集 cs.CR

The paper is submitted to "The 48th International ACM SIGIR  Conference on Research and Development in Information Retrieval" and is  currently under review

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09334v1) [paper-pdf](http://arxiv.org/pdf/2503.09334v1)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. All scripts required to reproduce the dataset, along with examples and relevant resources for replicating our results, will be made available upon the paper's acceptance.

摘要: 将大型语言模型(LLM)集成到网络安全应用程序中提供了重大机遇，如增强威胁分析和恶意软件检测，但也可能带来重大风险和安全问题，包括个人数据泄露和新恶意软件的自动生成。为了应对这些挑战，我们开发了CyberLLMInstruct，这是一个包含54,928个指令-响应对的数据集，涵盖了网络安全任务，如恶意软件分析、网络钓鱼模拟和零日漏洞。该数据集是通过多阶段过程构建的。这包括从多个来源获取数据，将其过滤和组织成指令-响应对，并使其与现实世界的情景相一致，以增强其适用性。选择了7个开源LLM来测试CyberLLMInstruct的有用性：Phi 3 Mini 3.8B、Mistral 7B、Qwen 2.5 7B、Llama 3 8B、Llama 3.1 8B、Gema 2 9B和Llama 2 70B。在我们的主要示例中，我们使用OWASP TOP 10框架严格评估微调模型的安全性，发现微调降低了所有测试的LLM和每个对手攻击的安全弹性(例如，针对即时注入的Llama 3.1 8B的安全分数从0.95下降到0.15)。在我们的第二个示例中，我们展示了这些相同的微调模型在CyberMetric基准上也可以达到高达92.50%的准确率。这些发现突出了性能和安全之间的权衡，显示了对抗性测试和进一步研究微调方法的重要性，这些方法可以降低安全风险，同时仍然提高不同数据集和域的性能。复制数据集所需的所有脚本，以及复制我们的结果的例子和相关资源，将在论文被接受时提供。



## **29. Prompt Inference Attack on Distributed Large Language Model Inference Frameworks**

对分布式大型语言模型推理框架的提示推理攻击 cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09291v1) [paper-pdf](http://arxiv.org/pdf/2503.09291v1)

**Authors**: Xinjian Luo, Ting Yu, Xiaokui Xiao

**Abstract**: The inference process of modern large language models (LLMs) demands prohibitive computational resources, rendering them infeasible for deployment on consumer-grade devices. To address this limitation, recent studies propose distributed LLM inference frameworks, which employ split learning principles to enable collaborative LLM inference on resource-constrained hardware. However, distributing LLM layers across participants requires the transmission of intermediate outputs, which may introduce privacy risks to the original input prompts - a critical issue that has yet to be thoroughly explored in the literature.   In this paper, we rigorously examine the privacy vulnerabilities of distributed LLM inference frameworks by designing and evaluating three prompt inference attacks aimed at reconstructing input prompts from intermediate LLM outputs. These attacks are developed under various query and data constraints to reflect diverse real-world LLM service scenarios. Specifically, the first attack assumes an unlimited query budget and access to an auxiliary dataset sharing the same distribution as the target prompts. The second attack also leverages unlimited queries but uses an auxiliary dataset with a distribution differing from the target prompts. The third attack operates under the most restrictive scenario, with limited query budgets and no auxiliary dataset available. We evaluate these attacks on a range of LLMs, including state-of-the-art models such as Llama-3.2 and Phi-3.5, as well as widely-used models like GPT-2 and BERT for comparative analysis. Our experiments show that the first two attacks achieve reconstruction accuracies exceeding 90%, while the third achieves accuracies typically above 50%, even under stringent constraints. These findings highlight privacy risks in distributed LLM inference frameworks, issuing a strong alert on their deployment in real-world applications.

摘要: 现代大型语言模型(LLM)的推理过程需要令人望而却步的计算资源，这使得它们不适合在消费级设备上部署。为了解决这一局限性，最近的研究提出了分布式LLM推理框架，该框架利用分裂学习原理在资源受限的硬件上实现协作式LLM推理。然而，在参与者之间分发LLM层需要传输中间输出，这可能会给原始输入提示带来隐私风险-这是一个尚未在文献中彻底探讨的关键问题。本文通过设计和评估三种旨在从中间LLM输出重构输入提示的提示推理攻击，严格检查了分布式LLM推理框架的隐私漏洞。这些攻击是在各种查询和数据约束下开发的，以反映现实世界中不同的LLM服务场景。具体地说，第一次攻击假定无限制的查询预算和对与目标提示共享相同分布的辅助数据集的访问。第二个攻击也利用无限查询，但使用的辅助数据集具有与目标提示不同的分布。第三种攻击是在限制最严格的情况下进行的，查询预算有限，并且没有可用的辅助数据集。我们在一系列LLM上评估了这些攻击，包括最先进的模型，如Llama-3.2和Phi-3.5，以及广泛使用的模型，如GPT-2和BERT，以进行比较分析。我们的实验表明，前两种攻击的重建精度超过90%，而第三种攻击的重建精度通常在50%以上，即使在严格的约束下也是如此。这些发现突出了分布式LLM推理框架中的隐私风险，对它们在现实世界应用程序中的部署发出了强烈的警告。



## **30. In-Context Defense in Computer Agents: An Empirical Study**

计算机代理中的上下文防御：实证研究 cs.AI

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09241v1) [paper-pdf](http://arxiv.org/pdf/2503.09241v1)

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou

**Abstract**: Computer agents powered by vision-language models (VLMs) have significantly advanced human-computer interaction, enabling users to perform complex tasks through natural language instructions. However, these agents are vulnerable to context deception attacks, an emerging threat where adversaries embed misleading content into the agent's operational environment, such as a pop-up window containing deceptive instructions. Existing defenses, such as instructing agents to ignore deceptive elements, have proven largely ineffective. As the first systematic study on protecting computer agents, we introduce textbf{in-context defense}, leveraging in-context learning and chain-of-thought (CoT) reasoning to counter such attacks. Our approach involves augmenting the agent's context with a small set of carefully curated exemplars containing both malicious environments and corresponding defensive responses. These exemplars guide the agent to first perform explicit defensive reasoning before action planning, reducing susceptibility to deceptive attacks. Experiments demonstrate the effectiveness of our method, reducing attack success rates by 91.2% on pop-up window attacks, 74.6% on average on environment injection attacks, while achieving 100% successful defenses against distracting advertisements. Our findings highlight that (1) defensive reasoning must precede action planning for optimal performance, and (2) a minimal number of exemplars (fewer than three) is sufficient to induce an agent's defensive behavior.

摘要: 由视觉语言模型(VLM)驱动的计算机代理极大地促进了人机交互，使用户能够通过自然语言指令执行复杂的任务。然而，这些代理容易受到上下文欺骗攻击，这是一种新兴的威胁，即对手在代理的操作环境中嵌入误导性内容，例如包含欺骗性指令的弹出窗口。现有的防御措施，如指示特工忽略欺骗性因素，已被证明在很大程度上无效。作为第一个关于保护计算机代理的系统研究，我们引入了Textbf[上下文中的防御}，利用上下文中的学习和思想链(COT)推理来对抗此类攻击。我们的方法涉及用一小组精心挑选的样本来增强代理的上下文，这些样本既包含恶意环境，也包含相应的防御响应。这些样本指导代理在行动计划之前首先执行明确的防御推理，降低对欺骗性攻击的敏感性。实验证明了该方法的有效性，对弹出窗口攻击的攻击成功率降低了91.2%，对环境注入攻击的平均成功率降低了74.6%，同时对分散注意力的广告实现了100%的成功防御。我们的发现强调：(1)为了获得最佳性能，防御推理必须先于行动计划，(2)最少的样本数量(少于3个)足以诱导代理人的防御行为。



## **31. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2410.23746v3) [paper-pdf](http://arxiv.org/pdf/2410.23746v3)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟各种提示用法、人工修改(如单词替换)和书写噪音(如拼写错误)。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **32. A Survey on Trustworthy LLM Agents: Threats and Countermeasures**

值得信赖的LLM代理人调查：威胁与对策 cs.MA

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09648v1) [paper-pdf](http://arxiv.org/pdf/2503.09648v1)

**Authors**: Miao Yu, Fanci Meng, Xinyun Zhou, Shilong Wang, Junyuan Mao, Linsey Pang, Tianlong Chen, Kun Wang, Xinfeng Li, Yongfeng Zhang, Bo An, Qingsong Wen

**Abstract**: With the rapid evolution of Large Language Models (LLMs), LLM-based agents and Multi-agent Systems (MAS) have significantly expanded the capabilities of LLM ecosystems. This evolution stems from empowering LLMs with additional modules such as memory, tools, environment, and even other agents. However, this advancement has also introduced more complex issues of trustworthiness, which previous research focused solely on LLMs could not cover. In this survey, we propose the TrustAgent framework, a comprehensive study on the trustworthiness of agents, characterized by modular taxonomy, multi-dimensional connotations, and technical implementation. By thoroughly investigating and summarizing newly emerged attacks, defenses, and evaluation methods for agents and MAS, we extend the concept of Trustworthy LLM to the emerging paradigm of Trustworthy Agent. In TrustAgent, we begin by deconstructing and introducing various components of the Agent and MAS. Then, we categorize their trustworthiness into intrinsic (brain, memory, and tool) and extrinsic (user, agent, and environment) aspects. Subsequently, we delineate the multifaceted meanings of trustworthiness and elaborate on the implementation techniques of existing research related to these internal and external modules. Finally, we present our insights and outlook on this domain, aiming to provide guidance for future endeavors.

摘要: 随着大型语言模型的快速发展，基于大型语言模型的智能体和多智能体系统极大地扩展了大型语言模型生态系统的能力。这种发展源于向LLM提供额外的模块，如内存、工具、环境，甚至其他代理。然而，这一进步也引入了更复杂的可信度问题，以前仅关注LLMS的研究无法涵盖这些问题。在本次调查中，我们提出了TrustAgent框架，这是一项关于代理可信性的综合研究，具有模块化分类、多维内涵和技术实现的特点。通过深入研究和总结新出现的针对代理和多代理系统的攻击、防御和评估方法，我们将可信LLM的概念扩展到新兴的可信代理范式。在TrustAgent中，我们从解构和引入代理和MAS的各种组件开始。然后，我们将他们的可信性分为内在(大脑、记忆和工具)和外在(用户、代理和环境)两个方面。随后，我们描述了可信性的多方面含义，并详细阐述了与这些内部和外部模块相关的现有研究的实现技术。最后，我们提出了我们对这一领域的见解和展望，旨在为未来的努力提供指导。



## **33. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

探索LLM中的潜在子空间以实现人工智能安全：识别和操纵敌对状态 cs.LG

4 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09066v1) [paper-pdf](http://arxiv.org/pdf/2503.09066v1)

**Authors**: Xin Wei Chia, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.

摘要: 大型语言模型(LLM)在各种任务中表现出了非凡的能力，但它们仍然容易受到对手操纵，例如通过即时注入攻击越狱。这些攻击绕过安全机制，生成受限或有害的内容。在这项研究中，我们通过从LLM中提取隐藏激活来研究安全状态和越狱状态的潜在潜在子空间。受神经科学中吸引子动力学的启发，我们假设LLM的激活稳定在半稳定状态，可以被识别和扰动以诱导状态转变。使用降维技术，我们从安全和越狱反应中投影激活，以揭示低维空间中的潜在子空间。然后我们推导出一个扰动向量，当应用于安全表示时，将模型转变为越狱状态。我们的结果表明，这种因果干预导致在提示的子集中有统计意义的越狱反应。接下来，我们探索这些扰动是如何通过模型的各层传播的，测试诱导的状态变化是保持局部化还是在整个网络中级联。我们的发现表明，有针对性的扰动导致激活和模型反应的明显转变。我们的方法为潜在的主动防御铺平了道路，从传统的基于护栏的方法转变为先发制人的、模型不可知的技术，在表征水平上中和敌对状态。



## **34. Battling Misinformation: An Empirical Study on Adversarial Factuality in Open-Source Large Language Models**

与错误信息作斗争：开源大型语言模型中对抗事实的实证研究 cs.CL

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.10690v1) [paper-pdf](http://arxiv.org/pdf/2503.10690v1)

**Authors**: Shahnewaz Karim Sakib, Anindya Bijoy Das, Shibbir Ahmed

**Abstract**: Adversarial factuality refers to the deliberate insertion of misinformation into input prompts by an adversary, characterized by varying levels of expressed confidence. In this study, we systematically evaluate the performance of several open-source large language models (LLMs) when exposed to such adversarial inputs. Three tiers of adversarial confidence are considered: strongly confident, moderately confident, and limited confidence. Our analysis encompasses eight LLMs: LLaMA 3.1 (8B), Phi 3 (3.8B), Qwen 2.5 (7B), Deepseek-v2 (16B), Gemma2 (9B), Falcon (7B), Mistrallite (7B), and LLaVA (7B). Empirical results indicate that LLaMA 3.1 (8B) exhibits a robust capability in detecting adversarial inputs, whereas Falcon (7B) shows comparatively lower performance. Notably, for the majority of the models, detection success improves as the adversary's confidence decreases; however, this trend is reversed for LLaMA 3.1 (8B) and Phi 3 (3.8B), where a reduction in adversarial confidence corresponds with diminished detection performance. Further analysis of the queries that elicited the highest and lowest rates of successful attacks reveals that adversarial attacks are more effective when targeting less commonly referenced or obscure information.

摘要: 对抗性真实性是指敌手故意在输入提示中插入错误信息，其特征是表现出不同程度的自信。在这项研究中，我们系统地评估了几个开放源码的大型语言模型(LLM)在面对这种敌意输入时的性能。我们考虑了三个等级的对手自信：强烈自信、适度自信和有限自信。我们的分析包括8个LLMS：骆驼3.1(8B)、Phi 3(3.8B)、Qwen 2.5(7B)、DeepSeek-v2(16B)、Gemma2(9B)、Falcon(7B)、Mstrallite(7B)和LLaVA(7B)。实验结果表明，Llama3.1(8B)表现出较强的检测敌意输入的能力，而Falcon(7B)表现出相对较低的性能。值得注意的是，对于大多数模型，检测成功率随着对手信心的降低而提高；然而，对于大羊驼3.1(8B)和Phi 3(3.8B)，这一趋势相反，其中对手信心的降低与检测性能的降低相对应。对导致最高和最低成功率的查询的进一步分析表明，对抗性攻击在针对不太常被引用或晦涩难懂的信息时更有效。



## **35. JBFuzz: Jailbreaking LLMs Efficiently and Effectively Using Fuzzing**

JBFuzz：使用Fuzzing高效、有效地越狱LLM cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.08990v1) [paper-pdf](http://arxiv.org/pdf/2503.08990v1)

**Authors**: Vasudev Gohil

**Abstract**: Large language models (LLMs) have shown great promise as language understanding and decision making tools, and they have permeated various aspects of our everyday life. However, their widespread availability also comes with novel risks, such as generating harmful, unethical, or offensive content, via an attack called jailbreaking. Despite extensive efforts from LLM developers to align LLMs using human feedback, they are still susceptible to jailbreak attacks. To tackle this issue, researchers often employ red-teaming to understand and investigate jailbreak prompts. However, existing red-teaming approaches lack effectiveness, scalability, or both. To address these issues, we propose JBFuzz, a novel effective, automated, and scalable red-teaming technique for jailbreaking LLMs.   JBFuzz is inspired by the success of fuzzing for detecting bugs/vulnerabilities in software. We overcome three challenges related to effectiveness and scalability by devising novel seed prompts, a lightweight mutation engine, and a lightweight and accurate evaluator for guiding the fuzzer. Assimilating all three solutions results in a potent fuzzer that only requires black-box access to the target LLM. We perform extensive experimental evaluation of JBFuzz using nine popular and widely-used LLMs. We find that JBFuzz successfully jailbreaks all LLMs for various harmful/unethical questions, with an average attack success rate of 99%. We also find that JBFuzz is extremely efficient as it jailbreaks a given LLM for a given question in 60 seconds on average. Our work highlights the susceptibility of the state-of-the-art LLMs to jailbreak attacks even after safety alignment, and serves as a valuable red-teaming tool for LLM developers.

摘要: 大型语言模型已经显示出作为语言理解和决策工具的巨大前景，它们已经渗透到我们日常生活的各个方面。然而，它们的广泛使用也伴随着新的风险，例如通过一种名为越狱的攻击产生有害的、不道德的或攻击性内容。尽管LLM开发人员做出了广泛的努力，利用人类的反馈来调整LLM，但它们仍然容易受到越狱攻击。为了解决这个问题，研究人员经常使用红色团队来理解和调查越狱提示。然而，现有的红团队方法缺乏有效性、可伸缩性或两者兼而有之。为了解决这些问题，我们提出了JBFuzz，一种新的高效、自动化和可扩展的越狱红色团队技术。JBFuzz的灵感来自于Fuzing在检测软件错误/漏洞方面的成功。我们通过设计新的种子提示、轻量级的突变引擎和轻量级且准确的赋值器来指导模糊器，从而克服了与有效性和可伸缩性相关的三个挑战。吸收所有这三种解决方案的结果是一个强大的模糊，只需要黑匣子访问目标LLM。我们使用九个流行和广泛使用的LLM对JBFuzz进行了广泛的实验评估。我们发现，JBFuzz成功地越狱了所有LLM，涉及各种有害/不道德的问题，平均攻击成功率为99%。我们还发现JBFuzz是非常高效的，因为它平均在60秒内为给定的问题越狱。我们的工作突出了最先进的LLM即使在安全调整之后也容易受到越狱攻击，并且对于LLM开发人员来说是一个有价值的红团队工具。



## **36. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (oral)

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2410.18469v3) [paper-pdf](http://arxiv.org/pdf/2410.18469v3)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.

摘要: 最近的研究表明，大型语言模型(LLM)容易受到自动越狱攻击，在自动越狱攻击中，由附加到有害查询的算法编制的敌意后缀绕过安全对齐并触发意外响应。目前生成这些后缀的方法计算量大，攻击成功率(ASR)低，尤其是针对Llama2和Llama3等排列良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一个迭代的自我调整过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架大大降低了生成敌意后缀的计算代价，同时在各种开源LLM上实现了近100个ASR。此外，它对闭源模型表现出很强的攻击可转移性，尽管只在Llama3上进行了优化，但在GPT-3.5上达到了99 ASR，在GPT-4上达到了49 ASR。除了提高越狱能力，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全配准研究提供了有价值的见解。



## **37. Backtracking for Safety**

为了安全而回溯 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08919v1) [paper-pdf](http://arxiv.org/pdf/2503.08919v1)

**Authors**: Bilgehan Sel, Dingcheng Li, Phillip Wallis, Vaishakh Keshava, Ming Jin, Siddhartha Reddy Jonnalagadda

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks, but ensuring their safety and alignment with human values remains crucial. Current safety alignment methods, such as supervised fine-tuning and reinforcement learning-based approaches, can exhibit vulnerabilities to adversarial attacks and often result in shallow safety alignment, primarily focusing on preventing harmful content in the initial tokens of the generated output. While methods like resetting can help recover from unsafe generations by discarding previous tokens and restarting the generation process, they are not well-suited for addressing nuanced safety violations like toxicity that may arise within otherwise benign and lengthy generations. In this paper, we propose a novel backtracking method designed to address these limitations. Our method allows the model to revert to a safer generation state, not necessarily at the beginning, when safety violations occur during generation. This approach enables targeted correction of problematic segments without discarding the entire generated text, thereby preserving efficiency. We demonstrate that our method dramatically reduces toxicity appearing through the generation process with minimal impact to efficiency.

摘要: 大型语言模型(LLM)在各种任务中表现出了非凡的能力，但确保它们的安全性和与人类价值观的一致性仍然至关重要。当前的安全对齐方法，如监督微调和基于强化学习的方法，可能会表现出对对手攻击的脆弱性，并且往往导致浅层安全对齐，主要侧重于防止生成输出的初始令牌中的有害内容。虽然像重置这样的方法可以通过丢弃之前的令牌并重新启动生成过程来帮助从不安全的世代中恢复，但它们不太适合于解决细微差别的安全违规行为，如毒性，否则可能会在良性和漫长的世代中出现。在本文中，我们提出了一种新的回溯方法，旨在解决这些局限性。我们的方法允许模型在发电过程中发生安全违规时恢复到更安全的发电状态，而不一定是在开始时。这种方法能够在不丢弃整个生成的文本的情况下有针对性地纠正有问题的片段，从而保持效率。我们证明，我们的方法大大减少了在产生过程中出现的毒性，对效率的影响最小。



## **38. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

人类可读的对抗性预言：使用情境背景对LLM漏洞的调查 cs.CL

arXiv admin note: text overlap with arXiv:2407.14644

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.16359v2) [paper-pdf](http://arxiv.org/pdf/2412.16359v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous studies that uncovered vulnerabilities in large language models (LLMs) frequently employed nonsensical adversarial prompts. However, such prompts can now be readily identified using automated detection techniques. To further strengthen adversarial attacks, we focus on human-readable adversarial prompts, which are more realistic and potent threats. Our key contributions are (1) situation-driven attacks leveraging movie scripts as context to create human-readable prompts that successfully deceive LLMs, (2) adversarial suffix conversion to transform nonsensical adversarial suffixes into independent meaningful text, and (3) AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B.

摘要: 之前发现大型语言模型（LLM）漏洞的研究经常使用无意义的对抗提示。然而，现在可以使用自动检测技术轻松识别此类提示。为了进一步加强对抗性攻击，我们重点关注人类可读的对抗性提示，这些提示是更现实、更强大的威胁。我们的主要贡献是（1）情境驱动攻击利用电影剧本作为上下文来创建人类可读的提示，成功欺骗LLM，（2）对抗性后缀转换，将无意义的对抗性后缀转换为独立有意义的文本，以及（3）具有p核采样的Advancer，一种生成多样化、人类可读的对抗性后缀的方法，提高GPT-3.5和Gemma 7 B等模型的攻击功效。



## **39. Proactive Privacy Amnesia for Large Language Models: Safeguarding PII with Negligible Impact on Model Utility**

大型语言模型的主动隐私咨询：保护PRI，对模型实用性的影响可忽略不计 cs.CL

ICLR'25 Poster. Project page and code is available at  https://ppa-iclr2025.my.canva.site/

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.17591v2) [paper-pdf](http://arxiv.org/pdf/2502.17591v2)

**Authors**: Martin Kuo, Jingyang Zhang, Jianyi Zhang, Minxue Tang, Louis DiValentin, Aolin Ding, Jingwei Sun, William Chen, Amin Hass, Tianlong Chen, Yiran Chen, Hai Li

**Abstract**: With the rise of large language models (LLMs), increasing research has recognized their risk of leaking personally identifiable information (PII) under malicious attacks. Although efforts have been made to protect PII in LLMs, existing methods struggle to balance privacy protection with maintaining model utility. In this paper, inspired by studies of amnesia in cognitive science, we propose a novel approach, Proactive Privacy Amnesia (PPA), to safeguard PII in LLMs while preserving their utility. This mechanism works by actively identifying and forgetting key memories most closely associated with PII in sequences, followed by a memory implanting using suitable substitute memories to maintain the LLM's functionality. We conduct evaluations across multiple models to protect common PII, such as phone numbers and physical addresses, against prevalent PII-targeted attacks, demonstrating the superiority of our method compared with other existing defensive techniques. The results show that our PPA method completely eliminates the risk of phone number exposure by 100% and significantly reduces the risk of physical address exposure by 9.8% - 87.6%, all while maintaining comparable model utility performance.

摘要: 随着大型语言模型(LLM)的兴起，越来越多的研究已经认识到它们在恶意攻击下泄露个人身份信息(PII)的风险。虽然已经做出了努力来保护LLMS中的PII，但现有的方法难以平衡隐私保护和维护模型效用。受认知科学中关于健忘症的研究启发，我们提出了一种新的方法，即主动隐私健忘症(PPA)，在保护LLMS中的PII的同时保持其实用性。这种机制的工作原理是主动识别和忘记序列中与PII最密切相关的关键记忆，然后使用适当的替代记忆植入记忆以维持LLM的功能。我们在多个模型上进行评估，以保护常见的PII，如电话号码和物理地址，免受流行的PII目标攻击，展示了我们的方法与其他现有防御技术相比的优越性。结果表明，我们的PPA方法完全消除了100%的电话号码暴露风险，并显著降低了9.8%-87.6%的物理地址暴露风险，所有这些都保持了可比的模型实用性能。



## **40. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.16750v3) [paper-pdf](http://arxiv.org/pdf/2502.16750v3)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **41. Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation**

对话注入攻击：通过上下文操纵越狱LLM cs.CL

17 pages, 10 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08195v1) [paper-pdf](http://arxiv.org/pdf/2503.08195v1)

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.

摘要: 大型语言模型(LLM)在广泛的应用程序中显示出了重要的实用价值；然而，它们的部署受到安全漏洞的困扰，特别是越狱攻击。这些攻击通过精心编制敌意提示来操纵LLM生成有害或不道德的内容。虽然目前对越狱攻击的大部分研究都集中在单回合互动上，但在很大程度上忽视了历史对话对模型行为的影响。在本文中，我们介绍了一种新的越狱范例，对话注入攻击(DIA)，它利用对话历史来提高此类攻击的成功率。DIA在黑盒设置中运行，只需要访问聊天API或了解LLM的聊天模板。我们提出了两种构造对抗性历史对话的方法：一种是采用灰盒预填充攻击，另一种是利用延迟响应。我们的实验表明，DIA在包括Llama-3.1和GPT-40在内的最近的LLM上达到了最先进的攻击成功率。此外，我们还证明了DIA可以绕过5种不同的防御机制，突出了其健壮性和有效性。



## **42. Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models**

针对大型语言模型的多回合越狱攻击的推理增强对话 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.11054v4) [paper-pdf](http://arxiv.org/pdf/2502.11054v4)

**Authors**: Zonghao Ying, Deyue Zhang, Zonglei Jing, Yisong Xiao, Quanchen Zou, Aishan Liu, Siyuan Liang, Xiangzheng Zhang, Xianglong Liu, Dacheng Tao

**Abstract**: Multi-turn jailbreak attacks simulate real-world human interactions by engaging large language models (LLMs) in iterative dialogues, exposing critical safety vulnerabilities. However, existing methods often struggle to balance semantic coherence with attack effectiveness, resulting in either benign semantic drift or ineffective detection evasion. To address this challenge, we propose Reasoning-Augmented Conversation, a novel multi-turn jailbreak framework that reformulates harmful queries into benign reasoning tasks and leverages LLMs' strong reasoning capabilities to compromise safety alignment. Specifically, we introduce an attack state machine framework to systematically model problem translation and iterative reasoning, ensuring coherent query generation across multiple turns. Building on this framework, we design gain-guided exploration, self-play, and rejection feedback modules to preserve attack semantics, enhance effectiveness, and sustain reasoning-driven attack progression. Extensive experiments on multiple LLMs demonstrate that RACE achieves state-of-the-art attack effectiveness in complex conversational scenarios, with attack success rates (ASRs) increasing by up to 96%. Notably, our approach achieves ASRs of 82% and 92% against leading commercial models, OpenAI o1 and DeepSeek R1, underscoring its potency. We release our code at https://github.com/NY1024/RACE to facilitate further research in this critical domain.

摘要: 多轮越狱攻击通过在迭代对话中使用大型语言模型(LLM)来模拟真实世界的人类交互，暴露出关键的安全漏洞。然而，现有的方法往往难以在语义一致性和攻击有效性之间取得平衡，导致良性的语义漂移或无效的检测规避。为了应对这一挑战，我们提出了一种新的多轮越狱框架--推理增强对话，该框架将有害的查询重新定义为良性的推理任务，并利用LLMS强大的推理能力来妥协安全对齐。具体地说，我们引入了攻击状态机框架来系统地建模问题转换和迭代推理，确保跨多轮的连贯查询生成。在这个框架的基础上，我们设计了增益引导的探索、自我发挥和拒绝反馈模块，以保留攻击语义，提高有效性，并支持推理驱动的攻击进展。在多个LLM上的大量实验表明，RACE在复杂的会话场景中获得了最先进的攻击效率，攻击成功率(ASR)提高了96%。值得注意的是，我们的方法在领先的商业模型OpenAI o1和DeepSeek r1上分别获得了82%和92%的ASR，这突显了它的有效性。我们在https://github.com/NY1024/RACE上发布我们的代码，以促进在这一关键领域的进一步研究。



## **43. Safety Guardrails for LLM-Enabled Robots**

LLM支持机器人的安全护栏 cs.RO

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07885v1) [paper-pdf](http://arxiv.org/pdf/2503.07885v1)

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots.

摘要: 尽管将大型语言模型(LLM)集成到机器人学中释放了变革性的能力，但它也带来了重大的安全问题，从平均情况下的LLM错误(例如，幻觉)到对抗性的越狱攻击，这可能会在现实世界中产生有害的机器人行为。传统的机器人安全方法不能解决LLM的新脆弱性，而当前的LLM安全护栏忽略了机器人在动态真实环境中操作所带来的物理风险。在本文中，我们提出了一种两级护栏结构RoboGuard，以确保LLM支持的机器人的安全。RoboGuard首先通过使用信任根LLM将预定义的安全规则与机器人环境相关联，该LLM使用思想链(COT)推理来生成严格的安全规范，如时间逻辑约束。然后，RoboGuard使用时态逻辑控制合成来解决这些上下文安全规范和可能不安全的计划之间的潜在冲突，这确保了安全合规性，同时将违反用户偏好的程度降至最低。通过考虑最坏情况下的越狱攻击的大量模拟和真实世界实验，我们证明了RoboGuard在不影响安全计划的性能的情况下，将不安全计划的执行率从92%降低到2.5%以下。我们还证明了RoboGuard是资源高效的，对自适应攻击具有健壮性，并且通过使其信任根LLM能够执行CoT推理而显著增强。这些结果突显了RoboGuard在缓解安全风险和提高启用LLM的机器人可靠性方面的潜力。



## **44. PoisonedParrot: Subtle Data Poisoning Attacks to Elicit Copyright-Infringing Content from Large Language Models**

PoisonedParrot：针对从大型语言模型中获取侵犯版权内容的微妙数据中毒攻击 cs.LG

18 pages, 18 figures. Accepted at NAACL 2025

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07697v1) [paper-pdf](http://arxiv.org/pdf/2503.07697v1)

**Authors**: Michael-Andrei Panaitescu-Liess, Pankayaraj Pathmanathan, Yigitcan Kaya, Zora Che, Bang An, Sicheng Zhu, Aakriti Agrawal, Furong Huang

**Abstract**: As the capabilities of large language models (LLMs) continue to expand, their usage has become increasingly prevalent. However, as reflected in numerous ongoing lawsuits regarding LLM-generated content, addressing copyright infringement remains a significant challenge. In this paper, we introduce PoisonedParrot: the first stealthy data poisoning attack that induces an LLM to generate copyrighted content even when the model has not been directly trained on the specific copyrighted material. PoisonedParrot integrates small fragments of copyrighted text into the poison samples using an off-the-shelf LLM. Despite its simplicity, evaluated in a wide range of experiments, PoisonedParrot is surprisingly effective at priming the model to generate copyrighted content with no discernible side effects. Moreover, we discover that existing defenses are largely ineffective against our attack. Finally, we make the first attempt at mitigating copyright-infringement poisoning attacks by proposing a defense: ParrotTrap. We encourage the community to explore this emerging threat model further.

摘要: 随着大型语言模型(LLM)的功能不断扩展，它们的使用也变得越来越普遍。然而，正如许多正在进行的关于LLM生成的内容的诉讼所反映的那样，解决侵犯版权的问题仍然是一个巨大的挑战。在本文中，我们介绍了PoisonedParot：第一种隐蔽的数据中毒攻击，即使模型没有针对特定的受版权保护的材料进行直接训练，也会诱导LLM生成受版权保护的内容。PoisonedParot使用现成的LLM将受版权保护的文本的小片段集成到毒药样本中。尽管它很简单，在广泛的实验中进行了评估，但PoisonedParot在启动模型以生成受版权保护的内容方面出人意料地有效，没有明显的副作用。此外，我们发现现有的防御在很大程度上对我们的攻击无效。最后，我们首次尝试通过提出一种防御方法来减轻版权侵权中毒攻击：ParrotTrap。我们鼓励社区进一步探索这种新兴的威胁模式。



## **45. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

恐怖谷：从扁平的角度探索对抗性的稳健性 cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2405.16918v2) [paper-pdf](http://arxiv.org/pdf/2405.16918v2)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization, but is also related to adversarial robustness since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples in the context of relative flatness: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running, it runs into a flat uncanny valley where the label remains flipped. In extensive experiments, we observe this phenomenon across various model architectures and datasets, even for adversarially trained models. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, adversarial examples rarely reach truly flat regions. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We, therefore theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.

摘要: 损失曲面的平坦性不仅与泛化正相关，而且还与对抗稳健性有关，因为输入的扰动与权重的扰动是非线性相关的。在这篇文章中，我们实证分析了对抗性例子与相对平坦度之间的关系。我们在相对平坦的背景下观察到对抗性例子的一个特殊性质：在迭代的一阶白盒攻击中，围绕对抗性例子测量的损失曲面的平坦性首先变得更尖锐，直到标签被翻转，但如果我们继续攻击，它会进入一个平坦的可怕山谷，在那里标签仍然被翻转。在广泛的实验中，我们在各种模型体系结构和数据集上观察到了这种现象，甚至对于经过恶意训练的模型也是如此。我们的结果也扩展到大型语言模型(LLM)，但由于输入空间的离散性质和相对较弱的攻击，对抗性例子很少到达真正平坦的区域。最重要的是，这一现象表明，平坦性本身不能解释对抗健壮性，除非我们也能保证函数在示例周围的行为。因此，理论上，我们通过限定损失曲面的三阶导数，将相对平坦性与对手的稳健性联系起来，强调了平坦性与稳健性模型的低全局Lipschitz常数相结合的必要性。



## **46. Utilizing Jailbreak Probability to Attack and Safeguard Multimodal LLMs**

利用越狱概率攻击和保护多模式LLM cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06989v1) [paper-pdf](http://arxiv.org/pdf/2503.06989v1)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Deyue Zhang, Dongdong Yang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal contents. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on inputs to maximize jailbreak probability. To counteract attacks, we also propose two defensive methods: Jailbreak-Probability-based Finetuning (JPF) and Jailbreak-Probability-based Defensive Noise (JPDN), which minimizes jailbreak probability in the MLLM parameters and input space, respectively. Extensive experiments show that (1) JPA yields improvements (up to 28.38\%) under both white and black box settings compared to previous methods with small perturbation bounds and few iterations. (2) JPF and JPDN significantly reduce jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 近年来，多通道大语言模型(MLLMS)在理解多通道内容方面表现出了优越的能力。然而，他们仍然容易受到越狱攻击，这些攻击利用他们的安全调整中的弱点来产生有害的反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLMS能力的二进制分类是不合适的。从这一观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表了当提示该输入时MLLMS生成恶意响应的可能性。我们通过对MLLMS的多次查询来近似这一概率。在使用越狱概率预测网络(JPPN)对输入隐藏状态与其对应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体地说，我们提出了基于越狱概率的攻击(JPA)，它优化了输入上的敌意扰动，以最大化越狱概率。为了对抗攻击，我们还提出了两种防御方法：基于越狱概率的精调(JPF)和基于越狱概率的防御噪声(JPDN)，它们分别在MLLM参数和输入空间中最小化越狱概率。大量的实验表明：(1)JPA在白盒和黑盒设置下都比以往的方法在小的扰动界和很少的迭代次数下都有很大的改善(高达28.38%)。(2)JPF和JPDN可显著减少越狱次数，最多可减少60%以上。以上两个结果都证明了引入越狱概率来细微区分不同输入越狱能力的意义。



## **47. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

InferDPT：黑匣子大型语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2310.12214v7) [paper-pdf](http://arxiv.org/pdf/2310.12214v7)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Zhikun Zhang

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.

摘要: 大型语言模型(LLM)，如ChatGPT，极大地简化了文本生成任务。然而，他们也对数据泄露和未经授权的数据收集等隐私风险表示担忧。现有的隐私保护推理解决方案面临着与计算时间和通信成本相关的实际挑战。在本文中，我们提出了第一个实用的黑盒LLMS隐私保护推理框架InferDPT，在文本生成中实现了差分隐私。InferDPT包括两个关键模块：“扰动模块”利用指数机制生成扰动提示，便于使用黑盒LLMS进行隐私保护推理；“提取模块”受知识提炼和检索-增强生成的启发，从扰动生成结果中提取连贯一致的文本，确保文本生成成功完成。针对以往指数机制易受修改攻击的隐私性问题，引入了一种新的差异化隐私机制RANTEXT，该机制集成在InferDPT的扰动模块中，引入了随机邻接的概念来处理提示内的文本扰动。在三个数据集上的实验结果表明，InferDPT的文本生成质量与非私有GPT-4相当，RANTEXT在隐私和效用之间的权衡方面超过了现有的最新机制SanText+和CUSTEXT+。即使在隐私参数epsilon值为6.0的情况下，RANTEXT对嵌入修改攻击的平均隐私保护率也超过90%，比SanText+高0.58倍，比CUSTEXT+高3.35倍。



## **48. Stepwise Reasoning Error Disruption Attack of LLMs**

LLM的逐步推理错误中断攻击 cs.AI

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2412.11934v3) [paper-pdf](http://arxiv.org/pdf/2412.11934v3)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications.

摘要: 大语言模型在复杂的推理任务中取得了显著的进展，但其在推理过程中的安全性和稳健性仍未得到充分的研究。现有的对LLM推理的攻击受到特定环境或缺乏不可见性的限制，限制了它们的可行性和普适性。为了应对这些挑战，我们提出了逐步推理错误中断(SEED)攻击，它巧妙地在先前的推理步骤中注入错误，以误导模型产生不正确的后续推理和最终答案。与以往的方法不同，SEED兼容零射和少射设置，保持了自然的推理流程，在不修改指令的情况下确保了隐蔽的执行。在四个不同模型的四个数据集上的广泛实验证明了SEED的有效性，揭示了LLMS在推理过程中对中断的脆弱性。这些发现强调了需要更多地关注LLM推理的健壮性，以确保在实际应用中的安全性。



## **49. Can Watermarking Large Language Models Prevent Copyrighted Text Generation and Hide Training Data?**

对大型语言模型进行水印可以阻止受版权保护的文本生成并隐藏训练数据吗？ cs.LG

19 pages, 7 figures. Published at AAAI 2025. Code will be available  at https://github.com/michael-panaitescu/watermark_copyright_aaai25

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2407.17417v2) [paper-pdf](http://arxiv.org/pdf/2407.17417v2)

**Authors**: Michael-Andrei Panaitescu-Liess, Zora Che, Bang An, Yuancheng Xu, Pankayaraj Pathmanathan, Souradip Chakraborty, Sicheng Zhu, Tom Goldstein, Furong Huang

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in generating diverse and contextually rich text. However, concerns regarding copyright infringement arise as LLMs may inadvertently produce copyrighted material. In this paper, we first investigate the effectiveness of watermarking LLMs as a deterrent against the generation of copyrighted texts. Through theoretical analysis and empirical evaluation, we demonstrate that incorporating watermarks into LLMs significantly reduces the likelihood of generating copyrighted content, thereby addressing a critical concern in the deployment of LLMs. However, we also find that watermarking can have unintended consequences on Membership Inference Attacks (MIAs), which aim to discern whether a sample was part of the pretraining dataset and may be used to detect copyright violations. Surprisingly, we find that watermarking adversely affects the success rate of MIAs, complicating the task of detecting copyrighted text in the pretraining dataset. These results reveal the complex interplay between different regulatory measures, which may impact each other in unforeseen ways. Finally, we propose an adaptive technique to improve the success rate of a recent MIA under watermarking. Our findings underscore the importance of developing adaptive methods to study critical problems in LLMs with potential legal implications.

摘要: 大型语言模型(LLM)在生成丰富多样的文本方面表现出了令人印象深刻的能力。然而，由于LLMS可能无意中产生了受版权保护的材料，因此出现了对侵犯版权的担忧。在这篇文章中，我们首先研究了水印LLM作为对版权文本产生的威慑的有效性。通过理论分析和实证评估，我们证明了在LLMS中加入水印显著降低了产生受版权保护内容的可能性，从而解决了LLMS部署中的一个关键问题。然而，我们也发现，水印可能会对成员关系推断攻击(MIA)产生意想不到的后果，MIA旨在识别样本是否属于预训练数据集，并可能用于检测侵犯版权的行为。令人惊讶的是，我们发现水印对MIA的成功率产生了不利影响，使得在预训练数据集中检测受版权保护的文本的任务变得更加复杂。这些结果揭示了不同监管措施之间复杂的相互作用，这些措施可能会以不可预见的方式相互影响。最后，我们提出了一种自适应技术来提高最近的MIA在水印下的成功率。我们的发现强调了开发适应性方法来研究LLMS中具有潜在法律含义的关键问题的重要性。



## **50. CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models in Retrieval-Augmented Language Generation**

CtrlRAG：检索增强语言生成中基于掩蔽语言模型的黑匣子对抗攻击 cs.CL

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06950v1) [paper-pdf](http://arxiv.org/pdf/2503.06950v1)

**Authors**: Runqi Sui

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by integrating external knowledge bases. However, this integration introduces a new security threat: adversaries can exploit the retrieval mechanism to inject malicious content into the knowledge base, thereby influencing the generated responses. Based on this attack vector, we propose CtrlRAG, a novel attack method designed for RAG system in the black-box setting, which aligns with real-world scenarios. Unlike existing attack methods, CtrlRAG introduces a perturbation mechanism using Masked Language Model (MLM) to dynamically optimize malicious content in response to changes in the retrieved context. Experimental results demonstrate that CtrlRAG outperforms three baseline methods in both Emotional Manipulation and Hallucination Amplification objectives. Furthermore, we evaluate three existing defense mechanisms, revealing their limited effectiveness against CtrlRAG and underscoring the urgent need for more robust defenses.

摘要: 检索-增强生成(RAG)系统通过集成外部知识库来增强大型语言模型(LLMS)。然而，这种集成带来了新的安全威胁：攻击者可以利用检索机制将恶意内容注入知识库，从而影响生成的响应。基于该攻击向量，我们提出了一种新的针对黑盒环境下RAG系统的攻击方法CtrlRAG，该方法符合实际场景。与现有的攻击方法不同，CtrlRAG引入了一种使用掩蔽语言模型(MLM)的扰动机制来动态优化恶意内容，以响应检索到的上下文的变化。实验结果表明，CtrlRAG在情绪操纵和幻觉放大目标上都优于三种基线方法。此外，我们评估了三种现有的防御机制，揭示了它们对CtrlRAG的有限有效性，并强调了对更强大防御的迫切需要。



