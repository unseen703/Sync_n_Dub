# #1 first idea is to divide pipeline to multiple stages.
# # It will first perform all the steps except for the lip syncing 

# then display its output to the user.
# outputs will be 
# total speakers in the clip, video category etc.
# along with their croped faces, detected gender, tone,  assigned voice,  original voice sample and dubbed voiece sample. they can ask agent to correct any of this part.

# then they will have access to dubbed video with ASD bbox faces marked and option to assign/correct the user based on the vidoe they see.
# SO how they will observe bbox marked video, and along with it they will see the transcript of the audio.

# at each part of video they can tag the time stamp and ask llm to change either, active speaker, modify the current transcript.
# then after the changes they can go ahead review changes and then start lip syncing.






changes should be shown just below the script changes in case of translation.
for gender and tone changes it should be visible against gender and tone mentioned just above the agent chat box. old gender/voice/tone should be strikethorughed and new should be visible as birght blue colour 


have edit agent page just below the main video and remove the edit agent page.

just below the original and lipsynced video speakers/voices/tone, agent chat box, script and lipsync start panel, prompt and applicable edit action should be there (entire edit page basically, with What this can change section removed.)

Faces fill about 26% of the frame height (threshold 25%), so this is a close-up and the extra resolution would show. MuseTalk suits it (256x256 crop, 8-21x clip length). This rule is calibrated on three clips from one video, so treat it as a starting point rather than an answer.
remove this lengthy explanation, just keep for larger face and multiple frames muse talk performs better, but for smaller face frame wave2lip woks better.

there is no neutral option in the tone/style drop down.


in the top where progress by is shown show entire pipeline including lipsync for all runs. till dub run show lipsync yet pending and after lip sync entire pipeline will be in complete status.

talking head genearation is not working showing insightface not found.

