#1 first idea is to divide pipeline to multiple stages.
# It will first perform all the steps except for the lip syncing 

then display its output to the user.
outputs will be 
total speakers in the clip, video category etc.
along with their croped faces, detected gender, tone,  assigned voice,  original voice sample and dubbed voiece sample. they can ask agent to correct any of this part.

then they will have access to dubbed video with ASD bbox faces marked and option to assign/correct the user based on the vidoe they see.
SO how they will observe bbox marked video, and along with it they will see the transcript of the audio.

at each part of video they can tag the time stamp and ask llm to change either, active speaker, modify the current transcript.
then after the changes they can go ahead review changes and then start lip syncing.
