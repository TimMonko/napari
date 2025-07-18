#!/usr/bin/env python3
"""
Script to check which GitHub contributors are missing from CITATION.cff

This script compares the contributors in the GitHub repository with those
listed in the CITATION.cff file to identify missing contributors.

Requirements:
- requests library for GitHub API calls
- pyyaml library for parsing CITATION.cff
- python-dotenv (optional) for GitHub token

Usage:
    python check_citation_contributors.py

Environment Variables:
    GITHUB_TOKEN: GitHub personal access token (optional, but recommended to avoid rate limits)
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import yaml


@dataclass
class Contributor:
    """Data class to hold contributor information"""
    login: str
    contributions: int
    type: str  # 'User' or 'Bot'

class GitHubContributorFetcher:
    """Fetches contributors from GitHub API"""

    def __init__(self, repo_owner: str, repo_name: str, token: Optional[str] = None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"

        # Set up headers with authentication if token provided
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'napari-citation-checker'
        }
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
    
    def get_contributors(self) -> list[Contributor]:
        """Fetch all contributors from the repository"""
        contributors = []
        page = 1
        per_page = 100
        
        print(f"Fetching contributors from {self.repo_owner}/{self.repo_name}...")
        
        while True:
            url = f"{self.base_url}/contributors"
            params = {
                'page': page,
                'per_page': per_page,
                'anon': 'false'  # Exclude anonymous contributors
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                
                # Check rate limit
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining < 10:
                        print(f"Warning: Only {remaining} API calls remaining")
                
                page_contributors = response.json()
                
                # If no contributors on this page, we're done
                if not page_contributors:
                    break
                
                # Filter out bots and add to our list
                for contrib in page_contributors:
                    if contrib['type'] == 'User':  # Skip bots
                        contributors.append(Contributor(
                            login=contrib['login'],
                            contributions=contrib['contributions'],
                            type=contrib['type']
                        ))
                
                print(f"Fetched page {page}: {len(page_contributors)} contributors")
                page += 1
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching contributors: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status: {e.response.status_code}")
                    print(f"Response text: {e.response.text}")
                sys.exit(1)
        
        print(f"Total contributors found: {len(contributors)}")
        return contributors

class CitationParser:
    """Parses CITATION.cff file to extract existing aliases"""
    
    def __init__(self, citation_file: Path):
        self.citation_file = citation_file
    
    def get_existing_aliases(self) -> set[str]:
        """Extract all aliases from the CITATION.cff file"""
        aliases = set()
        
        try:
            with open(self.citation_file, 'r', encoding='utf-8') as f:
                citation_data = yaml.safe_load(f)
            
            if 'authors' in citation_data:
                for author in citation_data['authors']:
                    if 'alias' in author and author['alias']:
                        # Handle both single alias and list of aliases
                        if isinstance(author['alias'], list):
                            aliases.update(author['alias'])
                        else:
                            aliases.add(author['alias'].strip())
            
            print(f"Found {len(aliases)} existing aliases in CITATION.cff")
            return aliases
            
        except FileNotFoundError:
            print(f"Error: CITATION.cff not found at {self.citation_file}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing CITATION.cff: {e}")
            sys.exit(1)

def main():
    """Main function to compare contributors and aliases"""
    
    # Configuration
    REPO_OWNER = "napari"
    REPO_NAME = "napari"
    CITATION_FILE = Path("CITATION.cff")
    
    # Get GitHub token from environment (optional but recommended)
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("Warning: No GITHUB_TOKEN environment variable set.")
        print("You may hit rate limits. Consider setting a GitHub personal access token.")
        print("See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token")
        print()
    
    # Parse existing aliases from CITATION.cff
    parser = CitationParser(CITATION_FILE)
    existing_aliases = parser.get_existing_aliases()
    
    # Fetch contributors from GitHub
    fetcher = GitHubContributorFetcher(REPO_OWNER, REPO_NAME, github_token)
    contributors = fetcher.get_contributors()
    
    # Get set of GitHub usernames
    github_usernames = {contrib.login for contrib in contributors}
    
    # Find missing contributors
    missing_contributors = github_usernames - existing_aliases
    
    # Find aliases that don't correspond to current contributors
    obsolete_aliases = existing_aliases - github_usernames
    
    # Sort contributors by contribution count for better reporting
    contributors_by_count = sorted(contributors, key=lambda x: x.contributions, reverse=True)
    missing_contributors_with_counts = [
        contrib for contrib in contributors_by_count 
        if contrib.login in missing_contributors
    ]

    # Print results
    print("\n" + "="*80)
    print("CITATION.CFF CONTRIBUTOR ANALYSIS")
    print("="*80)

    print(f"\nTotal GitHub contributors: {len(github_usernames)}")
    print(f"Total aliases in CITATION.cff: {len(existing_aliases)}")
    print(f"Missing from CITATION.cff: {len(missing_contributors)}")
    print(f"Obsolete aliases: {len(obsolete_aliases)}")
    
    if missing_contributors:
        print(f"\n🔍 MISSING CONTRIBUTORS ({len(missing_contributors)}):")
        print("-" * 50)
        for contrib in missing_contributors_with_counts:
            print(f"  {contrib.login:<30} ({contrib.contributions} contributions)")
    else:
        print("\n✅ All GitHub contributors are included in CITATION.cff!")
    
    if obsolete_aliases:
        print(f"\n⚠️  OBSOLETE ALIASES ({len(obsolete_aliases)}):")
        print("-" * 50)
        print("These aliases are in CITATION.cff but not in current GitHub contributors:")
        for alias in sorted(obsolete_aliases):
            print(f"  {alias}")
    
    # Generate a summary for easy copy-paste
    if missing_contributors:
        print(f"\n📋 SUMMARY FOR QUICK REFERENCE:")
        print("-" * 50)
        print("Missing GitHub usernames (sorted by contribution count):")
        usernames_only = [contrib.login for contrib in missing_contributors_with_counts]
        print(", ".join(usernames_only))
    
    print(f"\n💡 To add missing contributors, you'll need to:")
    print("1. Look up their real names and affiliations")
    print("2. Add them to the authors section in CITATION.cff")
    print("3. Include their GitHub username in the 'alias' field")
    
    return len(missing_contributors)

if __name__ == "__main__":
    try:
        missing_count = main()
        sys.exit(0 if missing_count == 0 else 1)
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
