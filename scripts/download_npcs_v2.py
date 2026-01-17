#!/usr/bin/env python3
"""
Download OSRS NPC data using Wiki's action API (more reliable)
"""
import json
import os
import requests
import time
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def download_npcs():
    """Download NPC/Monster data from OSRS Wiki"""
    print("Downloading NPCs via category members...")
    
    url = "https://oldschool.runescape.wiki/api.php"
    headers = {'User-Agent': 'OSRS-RL-Bot/1.0 (Training purposes)'}
    
    all_npcs = {}
    
    # Get category members (monster pages)
    cmcontinue = None
    page_titles = []
    
    print("Step 1: Getting monster page list...")
    while True:
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': 'Category:Monsters',
            'cmlimit': 500,
            'format': 'json'
        }
        if cmcontinue:
            params['cmcontinue'] = cmcontinue
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            data = resp.json()
            
            members = data.get('query', {}).get('categorymembers', [])
            for m in members:
                page_titles.append(m['title'])
            
            print(f"  Found {len(page_titles)} pages so far...")
            
            if 'continue' in data:
                cmcontinue = data['continue'].get('cmcontinue')
            else:
                break
                
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"\nStep 2: Fetching details for {len(page_titles)} monsters...")
    
    # Fetch in batches of 50
    batch_size = 50
    for i in range(0, len(page_titles), batch_size):
        batch = page_titles[i:i+batch_size]
        titles = '|'.join(batch)
        
        params = {
            'action': 'query',
            'titles': titles,
            'prop': 'revisions',
            'rvprop': 'content',
            'rvslots': 'main',
            'format': 'json'
        }
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            data = resp.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page in pages.items():
                if page_id == '-1':
                    continue
                    
                title = page.get('title', '')
                revisions = page.get('revisions', [])
                if not revisions:
                    continue
                
                content = revisions[0].get('slots', {}).get('main', {}).get('*', '')
                
                # Parse infobox
                npc_data = parse_monster_infobox(title, content)
                if npc_data:
                    all_npcs[title] = npc_data
            
            print(f"  Processed {min(i+batch_size, len(page_titles))}/{len(page_titles)}")
            time.sleep(1)
            
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            continue
    
    # Save
    filepath = os.path.join(DATA_DIR, 'npcs.json')
    with open(filepath, 'w') as f:
        json.dump(all_npcs, f, indent=2)
    
    print(f"\nSaved {len(all_npcs)} NPCs to {filepath}")
    return all_npcs


def parse_monster_infobox(title, content):
    """Extract data from monster infobox"""
    
    def extract_field(pattern, default=None):
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            # Clean wiki markup
            val = re.sub(r'\[\[([^|\]]+)\|?[^\]]*\]\]', r'\1', val)
            val = re.sub(r'\{\{[^}]+\}\}', '', val)
            val = val.strip()
            return val
        return default
    
    def extract_int(pattern, default=0):
        val = extract_field(pattern)
        if val:
            # Handle ranges like "1-5"
            nums = re.findall(r'\d+', str(val))
            if nums:
                return int(nums[0])
        return default
    
    combat = extract_int(r'\|combat\s*=\s*(\d+)')
    hitpoints = extract_int(r'\|hitpoints\s*=\s*(\d+)')
    max_hit = extract_int(r'\|max hit\s*=\s*([^\n|]+)')
    attack_style = extract_field(r'\|attack style\s*=\s*([^\n|]+)', 'Unknown')
    slayer_cat = extract_field(r'\|slayer cat\s*=\s*([^\n|]+)', '')
    slayer_xp = extract_int(r'\|slayer xp\s*=\s*([^\n|]+)')
    
    # Only include if it has some combat data
    if combat > 0 or hitpoints > 0:
        return {
            'name': title,
            'combat_level': combat,
            'hitpoints': hitpoints,
            'max_hit': max_hit,
            'attack_style': attack_style or 'Unknown',
            'slayer_category': slayer_cat or '',
            'slayer_xp': slayer_xp
        }
    
    return None


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_npcs()
