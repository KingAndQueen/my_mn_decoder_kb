import rdflib
import pdb
import my_graph
#from model import get_continuous_chunks
import re
def tokenize(sent):
	return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def graph_filter(): # filter the mid2name, only get the  code exist in KG
	f=my_graph.get_code_to_english('/data/analysis/qichuan_data/freebase/mid2name.txt')
	fbmap_code_eng=f[0]
	fb=rdflib.Graph('Sleepycat',identifier='freebase')
	fb.open('/data/analysis/qichuan_data/freebase/db')
	PREFIX='''
                PREFIX : <http://rdf.freebase.com/ns/>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        '''
	exist_code_list=set()
	for index,key in enumerate(fbmap_code_eng.keys()):
		print index
		_key=key.replace('/','.')
	#	print _key 
		if _key.index('.')==0:
			_key=_key[_key.index('.')+1:]
		else:
			print _key 
		sql_sub= 'ASK {<http://rdf.freebase.com/ns/'+_key+'> ?p ?o }' #can search object also
		sql_obj= 'ASK {?s ?p <http://rdf.freebase.com/ns/'+_key+'>}'
#		pdb.set_trace()
		result1=fb.query(PREFIX+sql_sub)
		result2=fb.query(PREFIX+sql_obj)
		if result1 or result2:
			exist_code_list.add(key)
	f=open('./useful_fb_code.txt','w')
	for key in exist_code_list:
		f.writelines(key+'\n')
	f.close()
	return exist_code_list

def map_file_filter(useful_list=set()):  # creat the new file with code existing in KG
	f=open('./useful_fb_code.txt','r')
	for line in f:
		useful_list.add(line.strip())
	f.close()
	g=open('/data/analysis/qichuan_data/freebase/mid2name.txt','r')
	outfile=open('./useful_Map.txt','w')
	pdb.set_trace()
	for line in g:
		key_value=line.split('\t')
		key=key_value[0].strip()
		if key in useful_list:
			outfile.writelines(line)
	g.close()
	outfile.close()
	
def change_entities_of_babi(outpath='',inpath='data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt',useful_Map='useful_Map.txt'):
	qa=open(inpath,'r')
	#code_eng,eng_code=my_graph.get_code_to_english(useful_Map)
	#babi_fb_map=dict()
	#pdb.set_trace()
	#for line in qa:
	#	word_set=get_continuous_chunks(line)
	#	for word in word_set:
	#		babi_fb_map[word]=None
	#for index,key in enumerate(babi_fb_map.keys()):
	#	babi_fb_map[key]=eng_code.keys()[index]
	#pdb.set_trace()
	babi_fb_map={'Daniel':'Daniel Conn',
			'John':'John Colliani',
			'Mary':'Mary Geneva Doud',
			'Sandra':'Sandra Burnhard'}
	new_qa=open(outpath,'w')
#	pdb.set_trace()
	for line in qa:
		word_set=tokenize(line)
		for word in word_set:
			new_entity=babi_fb_map.get(word)
			if new_entity:
				line=line.replace(word,new_entity)
		new_qa.writelines(line)
	new_qa.close()
	qa.close()
if __name__=='__main__':
	#exist_list=graph_filter()
	#map_file_filter()
	change_entities_of_babi(inpath='./my_data/qa2_two-supporting-facts_train.txt',outpath='./my_data/new_qa2_two-supporting-facts_train.txt')
	pdb.set_trace()
