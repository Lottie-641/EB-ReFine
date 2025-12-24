log = {
        'helpdesk': {'event_attribute': ['activity', 'resource', 'timesincecasestart','servicelevel','servicetype','workgroup','product','customer'], 'trace_attribute': ['supportsection','responsiblesection'],
                     'event_template': '{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace. {{workgroup}} managed the request for the {{product}} of {{customer}} with service {{servicetype}} of level {{servicelevel}}.',
                     'trace_template': 'Section {{supportsection}} led by {{responsiblesection}}', 'target':'activity'},


        'sepsis': {'event_attribute': ['activity','orggroup','timesincecasestart', 'Leucocytes','CRP','LacticAcid'], 'trace_attribute': ['InfectionSuspected','DiagnosticBlood','DisfuncOrg','SIRSCritTachypnea','Hypotensie','SIRSCritHeartRate','Infusion','DiagnosticArtAstrup','Age','DiagnosticIC','DiagnosticSputum','DiagnosticLiquor','DiagnosticOther','SIRSCriteria2OrMore','DiagnosticXthorax','SIRSCritTemperature','DiagnosticUrinaryCulture','SIRSCritLeucos','Oligurie','DiagnosticLacticAcid','Diagnose','Hypoxie','DiagnosticUrinarySediment','DiagnosticECG'],
                   'event_template': 'Org{{orggroup}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace. Leucocytes {{Leucocytes}} CRP {{CRP}} LacticAcid {{LacticAcid}}.',
                   'trace_template': 'Patient with Age {{Age}} clinic status: InfectionSuspected {{InfectionSuspected}} DiagnosticBlood {{DiagnosticBlood}} DisfuncOrg {{DisfuncOrg}} SIRSCritTachypnea {{SIRSCritTachypnea}} Hypotensie {{Hypotensie}} SIRSCritHeartRate {{SIRSCritHeartRate}} Infusion {{Infusion}} DiagnosticArtAstrup {{DiagnosticArtAstrup}} DiagnosticIC {{DiagnosticIC}} DiagnosticSputum {{DiagnosticSputum}} DiagnosticLiquor {{DiagnosticLiquor}} DiagnosticOther {{DiagnosticOther}} SIRSCriteria2OrMore {{SIRSCriteria2OrMore}} DiagnosticXthorax {{DiagnosticXthorax}} SIRSCritTemperature {{SIRSCritTemperature}} DiagnosticUrinaryCulture {{DiagnosticUrinaryCulture}} SIRSCritLeucos {{SIRSCritLeucos}} Oligurie {{Oligurie}} DiagnosticLacticAcid {{DiagnosticLacticAcid}} Diagnose {{Diagnose}} Hypoxie {{Hypoxie}} DiagnosticUrinarySediment {{DiagnosticUrinarySediment}} DiagnosticECG {{DiagnosticECG}}.', 'target':'activity'},

        'bpic2020': {'event_attribute': ['activity','resource','timesincecasestart', 'Role'], 'trace_attribute': ['Org','Project','Task'],
                     'event_template': '{{resource}} with role {{Role}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace.',
                     'trace_template': '{{Org}} managed the {{Project}} for {{Task}}.', 'target':'activity'},

        'BPIC15_1': {'event_attribute': ['activity', 'resource','timesincecasestart', 'question', 'monitoringResource'], 'trace_attribute': ['parts', 'responsibleactor', 'lastphase', 'landregisterid', 'casestatus', 'sumleges'],
                     'event_template': 'Res{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace. The open question {{question}} concerned {{monitoringResource}}.',
                     'trace_template': 'The application concerned the status {{casestatus}} of the {{parts}} as part of {{lastphase}} in the project associated with LandRegisterID: {{landregisterid}} with {{sumleges}} and responsible {{responsibleactor}}.', 'target':'activity'},

        'bpic2017_o': {'event_attribute': ['activity', 'resource', 'timesincecasestart', 'action'], 'trace_attribute': ["MonthlyCost", "CreditScore", "FirstWithdrawalAmount", "OfferedAmount","NumberOfTerms"],
                       'event_template': '{{resource}} performed {{activity}} with status {{action}} {{timesincecasestart}} seconds ago from the beginning of the trace.',
                       'trace_template': 'The MonthlyCost {{MonthlyCost}} for the loan, determined based on the score {{CreditScore}}, calculated considering the FirstWithdrawalAmount {{FirstWithdrawalAmount}}, the OfferedAmount {{OfferedAmount}}, and the NumberOfTerms {{NumberOfTerms}}.','target': 'activity'},

        'mip': {'event_attribute': ['activity','resource','timesincecasestart','numsession','userid','turn','userutterance','chatbotresponse'], 'trace_attribute': [],
                'event_template': '{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace. In session {{numsession}} turn {{turn}} the user utterance was {{userutterance}} and chatbot response was {{chatbotresponse}}.',
                'trace_template': '', 'target':'activity'}
}



